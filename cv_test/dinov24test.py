import os
import json
import torch
import clip
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run CLIP/DINO evaluation with flexible paths")

    parser.add_argument('--query_data_path', type=str, required=True, help='Path to query_data.json')
    parser.add_argument('--passages_path', type=str, required=True, help='Path to passages.json')
    parser.add_argument('--m2kr_image_path', type=str, required=True, help='Path to infoseek or query images')
    parser.add_argument('--split_image_path', type=str, required=True, help='Path to split passage images')
    parser.add_argument('--dino_model_path', type=str, required=True, help='Path to pretrained DINOv2 model')
    parser.add_argument('--output_path', type=str, default='result_dino.json', help='Output result json file')

    return parser.parse_args()
import open_clip
args = parse_args()

# 加载 CLIP 模型及预处理函数
def model_init(model_name_or_path="ViT-L/14"):
    processor = AutoImageProcessor.from_pretrained(args.dino_model_path)
    model = AutoModel.from_pretrained(args.dino_model_path)
    model = model.to('cuda')
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    #tokenizer = open_clip.get_tokenizer('ViT-H/14')
    return model, processor
def get_features(model,input):
    input['pixel_values'] = input['pixel_values'].to('cuda')
    return model(**input).last_hidden_state.mean(dim=1)

# 初始化 CLIP 模型
model, preprocess = model_init()


# 数据集类：支持 "classic" 和 "default" 两种模式
class M2RKDataset(Dataset):
    def __init__(self, data_path, preprocessor, split="val", mode="default"):
        """
        Args:
            data_path (str): 数据集根目录，应包含 passages.json 和 query_data.json 以及图片文件。
            preprocessor (callable): 图像预处理函数（例如 transforms）。
            split (str): 数据集划分标识，目前未做区分，可预留。
            mode (str): 数据集模式，"classic" 或 "default"。
                        - classic 模式返回：passage_id, passage_content, image_list
                        - default 模式返回：query_id, combined_text, image
        """
        self.data_path = data_path
        self.preprocessor = preprocessor
        self.mode = mode

        # 加载 passages.json
        passages_path = os.path.join("", "passages.json")
        with open(passages_path, "r", encoding="utf-8") as f:
            self.passages = json.load(f)

        # 加载 query_data.json
        query_path = os.path.join("", "query_data.json")
        with open(query_path, "r", encoding="utf-8") as f:
            self.queries = json.load(f)

        if isinstance(self.queries, dict):
            self.queries = [self.queries]

        # 根据模式选择数据来源
        if mode == "classic":
            self.data = self.passages
        else:
            self.data = self.queries

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "classic":
            # classic 模式：返回 (passage_id, passage_content, image_list)
            item = self.data[idx]
            passage_id = item.get("passage_id")
            passage_content = item.get("passage_content")
            img_rel_path = item.get("page_screenshot")
            img_rel_path = os.path.splitext(img_rel_path)[0]

            img_path = os.path.join(args.split_image_path, img_rel_path)

            image_list = []
            if os.path.exists(img_path) and os.path.isdir(img_path):
                for img_file in os.listdir(img_path):
                    full_img_path = os.path.join(img_path, img_file)
                    try:
                        with Image.open(full_img_path).convert("RGB") as img:
                            image_list.append(img.convert("RGB"))
                    except OSError:
                        image_list.append(Image.new("RGB", (5, 5), color=(0, 0, 0)))
            else:
                image_list.append(Image.new("RGB", (5, 5), color=(0, 0, 0)))
            if len(image_list) == 0:
                image_list.append(Image.new("RGB", (5, 5), color=(0, 0, 0)))
            return passage_id, passage_content, image_list
        else:
            # default 模式：返回 (query_id, combined_text, image)
            item = self.data[idx]
            query_id = item.get("question_id")
            question = item.get("question")
            instruction = item.get("instruction")
            combined_text = f"{instruction} {question}"
            img_rel_path = item.get("img_path")
            img_path = os.path.join(self.data_path, img_rel_path)
            with Image.open(img_path).convert("RGB") as img:
                image = img.convert("RGB")
            return query_id, combined_text, image


# collate 函数，用于 classic 模式数据加载
def dc_image_classic(batch):
    passage_id, passage_content, image_seg_list = zip(*batch)
    return list(passage_id), list(passage_content), list(image_seg_list)


# collate 函数，用于 default 模式数据加载
def dc_query(batch):
    query_id, combined_text, image = zip(*batch)
    query_ids = list(query_id)
    images = list(image)
    combined_texts = list(combined_text)
    captions = [
        f"Try to find the document screenshot with the reference image and the instruction. Given the reference image <image> and with the instruction: {cap}. Describe the document?"
        for cap in combined_texts]
    return query_ids, images, captions


# 截取文本中的前300个词
def get_first_300_words(text):
    words = text.split()
    return ' '.join(words[:300])


# 获取所有候选组的图像及文本特征（使用 CLIP）
def get_all_image_tensors(model, preprocess, dataset):
    eval_dataloader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=dc_image_classic,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )

    all_pages_features = []  # 每个元素对应一个候选组的所有特征（图像特征 + 文本特征）
    all_passage_ids = []     # 每个候选组对应的 passage_id

    for batch in tqdm(eval_dataloader):
        passage_ids, passage_content, image_seg_list = batch
        passage_ids = list(passage_ids)

        for i in range(len(passage_ids)):
            candidate_images = image_seg_list[i]
            candidate_feats = []

            # 分批处理 candidate_images（每次处理 2 张）
            for start in range(0, len(candidate_images), 16):
                end = start + 2
                sub_images = candidate_images[start:end]
                #print(sub_images)
                sub_images_input = preprocess(images=sub_images, return_tensors="pt")
                with torch.no_grad():
                    image_fts_batch = get_features(model,sub_images_input)
                candidate_feats.extend(image_fts_batch.cpu())

            # 对候选组添加文本特征：截取 passage_content 的前300个词
            cap = [f"The content of the document is :{get_first_300_words(passage_content[i])}."]
            #with torch.no_grad():
                #cap_fts_batch = model.encode_text(clip.tokenize(cap).to('cuda'))
            #candidate_feats.extend(cap_fts_batch.cpu())

            all_pages_features.append(candidate_feats)
            all_passage_ids.append(passage_ids[i])
        #break
        torch.cuda.empty_cache()

    return all_pages_features, all_passage_ids

def eval2(model, preprocess, eval_dataset, all_pages_features, all_passage_ids):
    results = []

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=32,
        collate_fn=dc_query,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4
    )

    all_query_features = []
    all_qids = []

    for batch in tqdm(eval_dataloader):
        question_ids, images, questions = batch
        question_ids = list(question_ids)
        all_qids.extend(question_ids)

        images = preprocess(images=images, return_tensors="pt")
        #images = torch.stack(images)
        with torch.no_grad():
            query_fts = get_features(model,images)
        all_query_features.append(query_fts.cpu())
        #break
    all_query_features = torch.cat(all_query_features, dim=0).to('cpu')

    for i, question_id in enumerate(tqdm(all_qids)):
        query_feature = all_query_features[i]
        candidate_similarities = []

        # Determine which prefix to keep based on the question_id
        if 0 <= i <= 1399:
            valid_prefix = "A"
        elif 1400 <= i<= 4414:
            valid_prefix = "Q"
        else:
            valid_prefix = "B"

        # Only consider those passages that start with the valid prefix
        for candidate_idx, candidate_group in enumerate(all_pages_features):
            passage_id = all_passage_ids[candidate_idx]
            if passage_id.startswith(valid_prefix):
                candidate_group_tensor = torch.stack(
                    [page_feature.to(query_feature.device) for page_feature in candidate_group]
                )
                similarity = F.cosine_similarity(
                    query_feature.unsqueeze(0),
                    candidate_group_tensor
                )
                max_similarity = similarity.max().item()
                candidate_similarities.append((candidate_idx, max_similarity))

        # Sort by similarity descending
        candidate_similarities.sort(key=lambda x: x[1], reverse=True)
        # Take top 10 passages
        top10_passages = [all_passage_ids[candidate_idx] for candidate_idx, _ in candidate_similarities[:10]]

        # Record maximum similarity score for that query
        max_sim = candidate_similarities[0][1] if candidate_similarities else 0

        results.append({
            "question_id": question_id,
            "retrieved_passages": top10_passages,
            "max_similarity": max_sim
        })

    return results

# 评估函数：对每个 query 计算与所有候选组的最大余弦相似度，并返回 top10 的 passage_id 以及最大相似度值
def eval(model, preprocess, eval_dataset, all_pages_features, all_passage_ids):
    results = []

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=32,
        collate_fn=dc_query,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4
    )

    all_query_features = []
    all_qids = []

    for batch in tqdm(eval_dataloader):
        question_ids, images, questions = batch
        question_ids = list(question_ids)
        all_qids.extend(question_ids)

        images =[ preprocess(image).to('cuda') for image in images]
        images = torch.stack(images)
        with torch.no_grad():
            query_fts = model.encode_image(images)
        all_query_features.append(query_fts.cpu())

    all_query_features = torch.cat(all_query_features, dim=0).to('cpu')

    for i, question_id in enumerate(tqdm(all_qids)):
        query_feature = all_query_features[i]
        candidate_similarities = []

        for candidate_idx, candidate_group in enumerate(all_pages_features):

            candidate_group_tensor = torch.stack([page_feature.to(query_feature.device) for page_feature in candidate_group])
            similarity = F.cosine_similarity(query_feature.unsqueeze(0), candidate_group_tensor)
            max_similarity = similarity.max().item()
            candidate_similarities.append((candidate_idx, max_similarity))

        # 按相似度降序排序
        candidate_similarities.sort(key=lambda x: x[1], reverse=True)
        # 取前 10 个候选组对应的 passage_id
        top10_passages = [all_passage_ids[candidate_idx] for candidate_idx, _ in candidate_similarities[:10]]
        # 记录该 query 检索到的候选组中的最大相似度（即最高得分）
        max_sim = candidate_similarities[0][1] if candidate_similarities else 0

        results.append({
            "question_id": question_id,
            "retrieved_passages": top10_passages,
            "max_similarity": max_sim
        })

    return results


# 初始化数据集（请根据实际路径修改）
dataset = M2RKDataset(args.m2kr_image_path,preprocessor=None,split='val',mode = 'default')
dataset_classic = M2RKDataset(args.m2kr_image_path,preprocessor=None,split='val',mode = 'classic')

# 获取候选组的所有图像及文本特征
all_pages_features, all_passage_ids = get_all_image_tensors(model, preprocess, dataset_classic)

# 执行评估，获得每个 query 的 top10 检索结果及最大相似度
result = eval2(model, preprocess, dataset, all_pages_features, all_passage_ids)

# 将结果写入 JSON 文件
with open('result_dino.json', 'w') as jf:
    json.dump(result, jf, indent=4)
