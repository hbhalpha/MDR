from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Subset  # 引入 Subset
from tqdm import tqdm
from peft import PeftConfig, PeftModel
from gme_inference import GmeQwen2VL
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn

import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torch
Image.MAX_IMAGE_PIXELS = None  # 禁用解压炸弹检查
# from tqdm import tqdm
import json
from peft import PeftConfig, PeftModel
import deepspeed
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler,RandomSampler
import argparse

def get_first_300_words(text):
    """
    截取文本中的前300个词
    """
    # 分割成词并提取前300个词
    words = text.split()
    return ' '.join(words[:50])

def parse_args():
    parser = argparse.ArgumentParser(description="Train GME with LoRA and contrastive loss")

    # 模型路径与LoRA参数
    parser.add_argument('--gme_path', type=str, required=True,default="/home/share/linhaoqiang/MMIR/models--Alibaba-NLP--gme-Qwen2-VL-7B-Instruct/snapshots/d42eca5a540526cfa982a349724b24b25c12a95e")
    #parser.add_argument('--max_image_tokens', type=int, default=480)
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # 数据路径

    parser.add_argument('--m2kr_train_data_path', type=str,default="/home/share/linhaoqiang/MMIR/M2KR/images" ,required=True)
    #parser.add_argument('--subset_samples', type=int, default=0)
    parser.add_argument('--doc_data_path', type=str,default="/home/share/linhaoqiang/MMIR/MMdocIR/page_images/page_images" ,required=True)
    parser.add_argument('--clean_data_path', type=str ,required=True)

    parser.add_argument('--M2KR_passages_path')
    # 训练参数
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--num_train_epochs', type=int, default=1)

    return parser.parse_args()
args= parse_args()
def model_init(model_name_or_path="Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"):
    gme = GmeQwen2VL(args.gme_path,ft_mode = True,       max_image_tokens=480,)
    config = LoraConfig(
        #task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        #inference_mode=False,  # 训练模式
        r=args.lora_r,  # Lora 秩
        lora_alpha=args.lora_alpha,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=args.lora_dropout,  # Dropout 比例
        bias="none",
    )
    gme.base.model = get_peft_model(gme.base.model, config)
    #gme.base.model = PeftModel.from_pretrained(gme.base.model,'/home/share/linhaoqiang/MMIR/fine-tune/sft_infoseek_infonce_sft_1e-5')
    print("model loaded")
    for name, param in gme.base.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    # Print trainable parameters
    trainable_param_count = sum(param.numel() for param in gme.base.parameters() if param.requires_grad)
    print(f"Number of trainable parameters: {trainable_param_count}")

    return gme
gme = model_init()

def contrastive_loss(query_fts, pos_fts, neg_fts, temperature=0.07):

    #query_fts = query_fts.unsqueeze(0)  # 扩展为 [1, ft_dim]

    pos_sim = F.cosine_similarity(query_fts, pos_fts, dim=-1) / temperature  # [pos_num]

    neg_sim = F.cosine_similarity(query_fts, neg_fts, dim=-1) / temperature  # [neg_num]

    loss = 0
    for p_sim in pos_sim:
        exp_pos_sim = torch.exp(p_sim)  # 正例的相似度指数化
        exp_neg_sim = torch.exp(neg_sim).sum()  # 所有负例的相似度指数化并求和
        loss += -torch.log(exp_pos_sim / (exp_pos_sim + exp_neg_sim))

    loss = loss / pos_fts.size(0)

    return loss
def find_image_path(img_id, base_dir=args.m2kr_train_data_path):
    """根据 img_id 在指定目录下依次尝试 jpg/JPG/png/JPEG 扩展名，找到则返回完整路径，否则返回 None。"""
    # 这里定义需要尝试的扩展名顺序
    exts = ["jpg", "JPG", "png", "JPEG","jpeg","PNG"]
    #print(img_id)
    for ext in exts:
        img_path = os.path.join(base_dir, f"{img_id}")
        if os.path.exists(img_path):
            return img_path
class M2RKDataset(Dataset):
    def __init__(self, data_path, preprocessor, split="val", mode="default"):
        """
        Args:
            data_path (str): Root directory of the dataset. This should contain 'all_contents.json',
                             'querys.json', and the image files.
            preprocessor (callable): Image preprocessing function (e.g., transforms).
            split (str): Dataset split identifier (currently not used but can be reserved for future).
            mode (str): Dataset mode. "default" mode will return:
                        - query_id, f"{instruction} {question}", image, pos_contents, neg_contents
        """
        self.data_path = data_path
        self.preprocessor = preprocessor
        self.mode = mode

        # Load the contents and queries
        with open(args.clean_data_path, "r") as f:
            self.queries = json.load(f)
        M2KR_passages_path = os.path.join("", args.M2KR_passages_path)
        with open(M2KR_passages_path, "r", encoding="utf-8") as f:
            self.M2KR_passages = json.load(f)

        # Preprocess all contents for quick access by passage_id
        self.M2KR_content_dict = {content["passage_id"]: content["passage_content"] for content in self.M2KR_passages}
        #print(self.M2KR_content_dict)
    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        query_id = query['question_id']
        question = query['query']
        q_img_path = query['query_img_path']
        pos = [query['postive']]
        neg = query['negtives']
        type = query['type']
        #print(type)
        if type == "m2kr":
            # Query details
            pos_contents = [
                f"The main content of the document is :{get_first_300_words(self.M2KR_content_dict[pid])}. Please describe the document."
                for pid in pos]
            neg_contents = [
                f"The main content of the document is :{get_first_300_words(self.M2KR_content_dict[pid])}. Please describe the document."
                for pid in neg]

            # Load image
            #img_path = os.path.join("/home/share/linhaoqiang/MMIR/M2KR/images", f"{query['img_id']}.jpg")
            img_path = find_image_path(q_img_path)
            image = Image.open(img_path).convert("RGB")
            image_resized = image.resize((384, 384))
            instruction_question = f"Given the reference image and with the instruction: {question}. Try to find the document with the reference image and the instruction. "
            return query_id, instruction_question, image_resized, pos_contents[0], neg_contents,type
        else:
            # Query details
            doc_name = query['doc_name']
            pos_img = [Image.open(os.path.join(args.doc_data_path ,f"{doc_name}_{page_id}.jpg" )).convert('RGB') for page_id in pos]
            neg_img = [Image.open(os.path.join(args.doc_data_path ,f"{doc_name}_{page_id}.jpg" )).convert('RGB') for page_id in neg]
            return query_id, question, 0, pos_img[0], neg_img,type

def data_collector(batch):
     query_id, instruction_question, image, pos_contents, neg_contents,type = zip(*batch)
     query_id = list(query_id)
     query = list(instruction_question)
     image = list(image)
     pos_contents = list(pos_contents)
     neg_contents = list(neg_contents)
     type = list(type)
     captions = [
         f"{cap}. "
         for cap in query]

     return query_id,image,captions,pos_contents,neg_contents,type
import torch.distributed as dist

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='no',  # 不进行验证
    save_strategy='no',  # 不保存模型
    save_total_limit=2,  # 最多保存2个模型
    logging_dir='./logs',  # 日志存储路径
    logging_steps=1,  # 每 800 步记录日志
    per_device_train_batch_size=1,  # 每个设备的训练 batch size
    per_device_eval_batch_size=32,  # 每个设备的验证 batch size
    num_train_epochs=args.num_train_epochs,  # 训练的轮数
    #max_steps=150,
    learning_rate=args.learning_rate,  # 学习率
    weight_decay=0.01,  # 权重衰减
    fp16=True,  # 开启混合精度训练
    warmup_steps=25,  # 学习率预热的步数
    report_to="none",  # 禁用报告到 wandb 等
    gradient_accumulation_steps=args.gradient_accumulation_steps,  # 每累计128个小批次后更新一次梯度
    torch_empty_cache_steps = 1,  # 清空缓存步数
    #deepspeed="./ds_config.json"  # 使用Deepspeed进行分布式训练
)
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_counter = 0  # 初始化批次计数器

    def get_model_on_rank_0(self):
        """
        Returns the current model on the main process (local_rank=0).
        """
        # Check if the current process is the main one
        if self.is_world_process_zero():
            return self.model
        else:
            # Return None on other processes
            return None

    def get_train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=4,
            drop_last=False,

        )
    def compute_loss(self, model, inputs):
        # 自定义损失计算逻辑
        #torch.cuda.empty_cache()
        #model.vlm = model.vlm.to('cuda')
        #ref_images, target_images, ref_image_sizes,tar_image_sizes,ref_caps,target_cap,modification, image_langs, image_names, labels = inputs
        query_ids,images,captions,pos_contents,neg_contents,types = inputs
        #print(pos_contents)
        query_id = query_ids[0]
        image = images
        pos_content = pos_contents[0]
        neg_content = neg_contents[0]
        caption = captions[0]
        type = types[0]
        #print(type)
        #print(pos_contents)
        #with torch.no_grad():
        #device = model.device
        #print(type)
        if type == "doc_ir":
            #print(pos_contents)
            #print(caption)
            #query_fts = model.get_fused_embeddings(texts=caption)
            #print(caption)
            #with torch.no_grad():
            target_pos_fts = model.get_fused_embeddings(images=pos_contents)
            #print(caption)
            query_fts = model.get_fused_embeddings(texts=captions)
            # print(111)
            target_neg_fts = model.get_fused_embeddings(images=neg_contents[0])
        else:
            #with torch.no_grad():
            query_fts = model.get_fused_embeddings(images=images, texts=caption)
            #print(111)
            target_pos_fts = model.get_fused_embeddings(texts=pos_contents)
            #print(111)
            target_neg_fts = model.get_fused_embeddings(texts=neg_contents[0])
        batch_size = query_fts.size(0)
        device = query_fts.device
        loss = contrastive_loss(query_fts,pos_fts=target_pos_fts,neg_fts=target_neg_fts)
        if type == 'doc_ir':
            loss = loss
        loss = loss.to('cuda')
        #print(loss.device)
        return loss
    def training_step(self, *args, **kwargs):
        #torch.cuda.empty_cache()
        # 调用父类的 training_step 方法
        output = super().training_step(*args, **kwargs)

        # 增加批次计数器
        self.batch_counter += 1

        return output
gme.base.model = gme.base.model.train()
torch.distributed.init_process_group(backend='nccl',init_method='tcp://localhost:23457', world_size=1, rank=0)
dataset = M2RKDataset(data_path= "",preprocessor=None,split="train",mode="default")
#time.sleep(30)

import torch
from torch.utils.data import Subset
import random
# 获取数据集的总长度
dataset_size = len(dataset)
# 随机选择 200 个样本的索引
random_indices = random.sample(range(dataset_size), 100)
# 创建 Subset 数据集
subset_dataset = Subset(dataset, random_indices)

trainer = CustomTrainer(
    model=gme,
    args=training_args,
    train_dataset=dataset,
   data_collator=data_collector
)

trainer.train()
trainer.model.base.model.save_pretrained(f'{args.output_dir}')
training_args_dict = vars(training_args)  # 将 TrainingArguments 转换为字典
with open(f'{args.output_dir}/train_args.json', 'w') as f:
    json.dump(training_args_dict, f)
"""
python train_script.py \
  --gme_path "to/your_path/gme-Qwen2-VL-7B-Instruct" \
  --m2kr_train_data_path "to/your_path/images" \
  --doc_data_path "to/your_path/page_images" \
  --clean_data_path "to/your_path/clean_queries.json" \
  --M2KR_passages_path "to/your_path/passages.json" \
  --output_dir "./results/gme_lora" \
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1

"""