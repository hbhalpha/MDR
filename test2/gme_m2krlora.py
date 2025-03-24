import torch
import os
import json
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftConfig, PeftModel
from gme_inference import GmeQwen2VL
from torch.utils.data import Dataset
from PIL import Image

# Disabling parallel tokenization
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#python ./gme_m2krlora.py --model_name_or_path "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5" --gme_path "/path/to/gme" --peft_model_path "/path/to/peft_model" --dataset_path "/path/to/dataset"

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run Vision2Seq evaluation")

    # Add arguments for hyperparameters
    parser.add_argument('--model_name_or_path', type=str,
                        default="Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5", help='Model name or path')
    parser.add_argument('--gme_path', type=str,
                        default="/home/share/linhaoqiang/MMIR/models--Alibaba-NLP--gme-Qwen2-VL-7B-Instruct/snapshots/d42eca5a540526cfa982a349724b24b25c12a95e",
                        help='Path to GME model')
    parser.add_argument('--peft_model_path', type=str,
                        default="/home/share/linhaoqiang/MMIR/zhengliu/sft_mixlora64_infonce_sft_soft_1e-5_doc-32-shunxu",
                        help='Path to PEFT model')
    parser.add_argument('--dataset_path', type=str, default='/home/share/linhaoqiang/MMIR/M2KR/images',
                        help='Path to the dataset')
    parser.add_argument('--dataset_classic_path', type=str, default='/home/share/linhaoqiang/MMIR/M2KR/Challenge',
                        help='Path to the classic dataset')
    parser.add_argument('--query_path')
    parser.add_argument('--passage_path')
    parser.add_argument('--split', type=str, default='val', help='Split for dataset (e.g., train, val)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--top_k', type=int, default=10, help='Top-K retrieval results')
    parser.add_argument('--image_resize', type=int, default=720, help='Resize short side of image')
    parser.add_argument('--query_batch_size', type=int, default=16, help='Batch size for queries')
    parser.add_argument('--save_path', type=str, default='expert1', help='Batch size for queries')

    return parser.parse_args()


# Initialize the configuration from command line arguments
args = parse_args()


# Initialize model and tokenizer
def model_init(model_name_or_path=args.model_name_or_path):
    gme = GmeQwen2VL(args.gme_path)
    gme.base.model = PeftModel.from_pretrained(gme.base.model, args.peft_model_path)
    return gme


model = model_init()


class M2RKDataset(Dataset):
    def __init__(self, data_path, preprocessor, split="val", mode="default"):
        self.data_path = data_path
        self.preprocessor = preprocessor
        self.mode = mode
        passages_path = os.path.join(data_path, args.passage_path)
        with open(passages_path, "r", encoding="utf-8") as f:
            self.passages = json.load(f)

        query_path = os.path.join(data_path, args.query_path)
        with open(query_path, "r", encoding="utf-8") as f:
            self.queries = json.load(f)

        if isinstance(self.queries, dict):
            self.queries = [self.queries]

        if mode == "classic":
            self.data = self.passages
        else:
            self.data = self.queries

    def __len__(self):
        return len(self.data)

    def process_image(self, image):
        orig_width, orig_height = image.size
        if max(orig_width, orig_height) > 10 * min(orig_width, orig_height):
            image = Image.new("RGB", (1, 1), color=(0, 0, 0))
            return [image]

        if max(orig_height, orig_width) < args.image_resize:
            return [image]

        if orig_width >= orig_height:
            scale = args.image_resize / orig_height
            new_height = args.image_resize
            new_width = int(round(orig_width * scale))
            orientation = 'landscape'
        else:
            scale = args.image_resize / orig_width
            new_width = args.image_resize
            new_height = int(round(orig_height * scale))
            orientation = 'portrait'

        resized_image = image.resize((new_width, new_height), resample=Image.ANTIALIAS)

        segments = []
        if orientation == 'landscape':
            seg_width = int(round(new_height * 1.41))
            if seg_width >= new_width:
                segments.append(resized_image)
            else:
                num_full_segments = new_width // seg_width
                for i in range(num_full_segments):
                    left = i * seg_width
                    right = left + seg_width
                    box = (left, 0, right, new_height)
                    seg = resized_image.crop(box)
                    segments.append(seg)
                if new_width % seg_width != 0:
                    left = num_full_segments * seg_width
                    box = (left, 0, new_width, new_height)
                    seg = resized_image.crop(box)
                    segments.append(seg)
        else:
            seg_height = int(round(new_width * 1.41))
            if seg_height >= new_height:
                segments.append(resized_image)
            else:
                num_full_segments = new_height // seg_height
                for i in range(num_full_segments):
                    top = i * seg_height
                    bottom = top + seg_height
                    box = (0, top, new_width, bottom)
                    seg = resized_image.crop(box)
                    segments.append(seg)
                if new_height % seg_height != 0:
                    top = num_full_segments * seg_height
                    box = (0, top, new_width, new_height)
                    seg = resized_image.crop(box)
                    segments.append(seg)

        return segments

    def __getitem__(self, idx):
        if self.mode == "classic":
            item = self.data[idx]
            passage_id = item.get("passage_id")
            passage_content = item.get("passage_content")
            img_rel_path = item.get("page_screenshot")
            img_path = os.path.join(self.data_path, img_rel_path)

            if not os.path.exists(img_path):
                image = Image.new("RGB", (1, 1), color=(0, 0, 0))
            else:
                try:
                    with Image.open(img_path) as img:
                        image = img.convert("RGB")
                except OSError as e:
                    print(f"Error loading image {img_path}: {e}")
                    image = Image.new("RGB", (1, 1), color=(0, 0, 0))

            return passage_id, passage_content, image
        else:
            item = self.data[idx]
            query_id = item.get("question_id")
            question = item.get("question")
            instruction = item.get("instruction")
            combined_text = f"{instruction} {question}"

            img_rel_path = item.get("img_path")
            img_path = os.path.join(self.data_path, img_rel_path)
            with Image.open(img_path) as img:
                image = img.convert("RGB")

            return query_id, combined_text, image


def eval2(model, eval_dataset, all_pages_features: torch.Tensor, all_passage_ids: list):
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=dc_query,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers
    )

    all_query_features = []
    all_qids = []

    for batch in tqdm(eval_dataloader, desc="Encoding queries"):
        question_ids, images, questions = batch

        question_ids = list(question_ids)
        all_qids.extend(question_ids)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                query_fts = model.get_fused_embeddings(images=images, texts=questions)

        all_query_features.append(query_fts.cpu())

    all_query_features = torch.cat(all_query_features, dim=0)
    all_pages_features = all_pages_features.cpu()
    all_query_features = all_query_features.cpu()

    A_idx, Q_idx, B_idx = [], [], []
    for i, pid in enumerate(all_passage_ids):
        if pid.startswith("A"):
            A_idx.append(i)
            continue
        elif pid.startswith("Q"):
            Q_idx.append(i)
        else:
            B_idx.append(i)

    A_pages = all_pages_features[A_idx] if A_idx else torch.empty(0, all_pages_features.size(1))
    Q_pages = all_pages_features[Q_idx] if Q_idx else torch.empty(0, all_pages_features.size(1))
    B_pages = all_pages_features[B_idx] if B_idx else torch.empty(0, all_pages_features.size(1))

    A_passage_ids = [all_passage_ids[i] for i in A_idx]
    Q_passage_ids = [all_passage_ids[i] for i in Q_idx]
    B_passage_ids = [all_passage_ids[i] for i in B_idx]

    A_q_indices, Q_q_indices, B_q_indices = [], [], []
    for i, qid in enumerate(all_qids):
        if 0 <= i <= 1399:
            A_q_indices.append(i)
        elif 1400 <= i <= 4414:
            Q_q_indices.append(i)
        else:
            B_q_indices.append(i)

    A_queries = all_query_features[A_q_indices] if A_q_indices else torch.empty(0, all_query_features.size(1))
    Q_queries = all_query_features[Q_q_indices] if Q_q_indices else torch.empty(0, all_query_features.size(1))
    B_queries = all_query_features[B_q_indices] if B_q_indices else torch.empty(0, all_query_features.size(1))

    results = [None] * len(all_qids)

    def compute_topk_and_store(query_feats: torch.Tensor,
                               passage_feats: torch.Tensor,
                               passage_ids: list,
                               query_indices: list):
        if query_feats.size(0) == 0 or passage_feats.size(0) == 0:
            for row_idx in query_indices:
                qid = all_qids[row_idx]
                results[row_idx] = {
                    'question_id': qid,
                    'retrieved_passages': []
                }
            return

        similarity_scores = torch.matmul(query_feats, passage_feats.T)
        _, topk_indices = torch.topk(similarity_scores, k=args.top_k, dim=1)

        for j, row_idx in enumerate(query_indices):
            qid = all_qids[row_idx]
            top_passages = [passage_ids[idx] for idx in topk_indices[j]]
            results[row_idx] = {
                'question_id': qid,
                'retrieved_passages': top_passages
            }

    compute_topk_and_store(A_queries, A_pages, A_passage_ids, A_q_indices)
    compute_topk_and_store(Q_queries, Q_pages, Q_passage_ids, Q_q_indices)
    compute_topk_and_store(B_queries, B_pages, B_passage_ids, B_q_indices)

    return results

def dc_image_classic(batch):
    passage_id, passage_content, image_seg_list= zip(*batch)
    passage_id = list(passage_id)
    passage_content = list(passage_content)
    image_seg_list = list(image_seg_list)
    return passage_id,passage_content,image_seg_list
def dc_query(batch):
    query_id, combined_text, image= zip(*batch)
    query_ids = list(query_id)
    images = list(image)
    combined_texts = list(combined_text)
    # print(captions)
    # print(image_names)
    #captions = list(captions)
    #image_langs = ["Describe the image <image> ? <|end|> <|end|> <|end|> <|end|> <|end|>"] * len(captions)
    captions = [
        f"Given the reference image and with the instruction: {cap}. Try to find the document with the reference image and the instruction. "
        for cap in combined_texts]
    return query_ids,images,captions



def get_all_image_tensors(model, dataset):
    """
    Extract visual features for each candidate group (passage) from the dataset.

    Args:
        model: The model used to extract visual features.
        dataset: The dataset, where each item is a tuple:
                 (passage_id, passage_content, image_seg_list, image_seg_size).

    Returns:
        all_pages_features: A list of lists. Each inner list contains multiple page features (tensors)
                            for one candidate group.
        passage_ids: A list of passage IDs corresponding to each candidate group.
    """
    # Initialize DataLoader for the dataset with the collate function dc_image.
    eval_dataloader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=dc_image_classic,
        # dc_image_classic 需提前定义，确保返回 (passage_ids, passage_contents, image_seg_list, image_seg_size)
        shuffle=False,
        drop_last=False,
        num_workers=4,  # 根据 CPU 核心数设置合适的 worker 数量
        pin_memory=True  # 可选项，当使用 GPU 时可加速数据传输
    )

    all_pages_features = []  # 每个元素对应一个候选组的所有图片特征
    all_passage_ids = []  # 每个候选组的 passage_id
    k=0
    # 遍历 DataLoader 中的每个批次
    for batch in tqdm(eval_dataloader):
        k+=1
        # 解包批次数据：假设 batch 返回 (passage_ids, passage_contents, image_seg_list, image_seg_size)
        passage_ids, passage_content, image_list = batch
        #if k <= 12:
            #continue
        passage_ids = list(passage_ids)
        # 注意：这里假设 image_seg_list 和 image_seg_size 均为列表，每个元素对应一个候选组中的所有图像分割
        # 如果 image_seg_list[i] 不是列表，则将其包装为列表
        # 同理，image_seg_size[i] 也应为列表
        cap = [
                  f"The content of the document is :{get_first_300_words(passage)}." for passage in passage_content]
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                image_fts_batch = model.get_text_embeddings(texts=cap)
        all_pages_features.append(image_fts_batch)
        all_passage_ids.extend(passage_ids)
        torch.cuda.empty_cache()
        #break
    all_pages_features = torch.cat(all_pages_features, dim=0)
    return all_pages_features, all_passage_ids
def get_first_300_words(text):
    """
    截取文本中的前300个词
    """
    # 分割成词并提取前300个词
    words = text.split()
    return ' '.join(words[:50])

# Now use the configurable model
dataset = M2RKDataset(args.dataset_path, preprocessor=None, split=args.split, mode='default')
dataset_classic = M2RKDataset(args.dataset_classic_path, preprocessor=None, split=args.split, mode='classic')
all_pages_features, all_passage_ids = get_all_image_tensors(model, dataset_classic)
result = eval2(model, dataset, all_pages_features, all_passage_ids)

with open(args.save_path, 'w') as jf:
    json.dump(result, jf, indent=4)
