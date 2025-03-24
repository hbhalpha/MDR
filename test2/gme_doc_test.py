import random
import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from peft import PeftConfig, PeftModel
from gme_inference import GmeQwen2VL
import os
import argparse
import json
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F

#python script.py --gme_path "/path/to/gme" --peft_model_path "/path/to/peft_model" --data_path "/path/to/data" --passages_file "/path/to/passages.json" --queries_file "/path/to/queries.json" --batch_size 2 --top_k 10

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run Vision2Seq evaluation")

    # Model and data paths
    parser.add_argument('--gme_path', type=str, required=True, help='Path to the GME model.')
    parser.add_argument('--peft_model_path', type=str, required=True, help='Path to the PEFT model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset.')
    parser.add_argument('--passages_file', type=str, required=True, help='Path to the passages file.')
    parser.add_argument('--queries_file', type=str, required=True, help='Path to the queries file.')
    parser.add_argument('--image_resize', type=int, default=720, help='Resize the short side of the image.')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for evaluation.')
    parser.add_argument('--top_k', type=int, default=10, help='Top-K results to retrieve for each query.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')

    return parser.parse_args()


# Parse command-line arguments
args = parse_args()


# Model initialization
def model_init():
    gme = GmeQwen2VL(args.gme_path)
    config = LoraConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=64,  # Lora rank
        lora_alpha=16,  # Lora alpha
        lora_dropout=0.05,  # Dropout rate
    )
    gme.base.model = PeftModel.from_pretrained(gme.base.model, args.peft_model_path)
    return gme


model = model_init()


# Dataset class for MMDocIR dataset
class MMDocIR_dataset(Dataset):
    def __init__(self, data_path, preprocessor, mode=''):
        self.data_path = data_path
        self.preprocessor = preprocessor
        self.mode = mode

        # Load passages and queries from the given files
        with open(args.passages_file, 'r') as f:
            self.passages = json.load(f)

        with open(args.queries_file, 'r') as f:
            self.queries = json.load(f)

        # Prepare data based on mode
        if mode == 'image':
            self.data = self._prepare_image_data()
        elif mode == 'query':
            self.data = self._prepare_query_data()
        else:
            raise ValueError("Mode should be either 'image' or 'query'")

    def _prepare_image_data(self):
        image_data = []
        for doc_name, passages in self.passages.items():
            for passage in passages:
                passage_id = passage['passage_id']
                image_path = passage['image_path']
                image_data.append((image_path, passage_id, doc_name))
        return image_data

    def _prepare_query_data(self):
        query_data = []
        for query in self.queries:
            question_id = query['question_id']
            question = query['question']
            doc_name = query['doc_name']
            query_data.append((question_id, question, doc_name))
        return query_data

    def __len__(self):
        return len(self.data)

    def _load_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        return image

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.mode == 'image':
            image_path, page_number, doc_name = item
            image = self._load_image(os.path.join(self.data_path, image_path))
            return image, page_number, doc_name
        elif self.mode == 'query':
            question_id, question, doc_name = item
            processed_question = question
            return question_id, processed_question, doc_name


# Dataloader collate functions
def dc_image(batch):
    images, page_number, doc_name = zip(*batch)
    images = list(images)
    page_number = list(page_number)
    doc_name = list(doc_name)
    image_langs = ["Describe the document screenshot"] * len(doc_name)
    return (images, image_langs, doc_name, page_number)


def dc_query(batch):
    question_id, processed_question, doc_name = zip(*batch)
    question_id = list(question_id)
    processed_question = list(processed_question)
    doc_name = list(doc_name)
    processed_question = [f"{q}" for q in processed_question]
    return (question_id, processed_question, doc_name)


# Evaluation function
def eval(model, eval_dataset, all_pages_features: dict):
    results = []
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=dc_query,
        shuffle=False,
        drop_last=False
    )

    for batch in tqdm(eval_dataloader):
        question_ids, questions, doc_names = batch
        question_ids = list(question_ids)

        with torch.cuda.amp.autocast():
            query_fts = model.get_text_embeddings(texts=questions)

        for i, question_id in enumerate(question_ids):
            query_feature = query_fts[i]
            doc_name = doc_names[i]
            page_features = all_pages_features[doc_name]
            page_similarities = []

            for page_id, page_feature in page_features.items():
                similarity = F.cosine_similarity(query_feature.unsqueeze(0), page_feature.unsqueeze(0))
                page_similarities.append((page_id, similarity.item()))

            page_similarities.sort(key=lambda x: x[1], reverse=True)
            top_k_passages = [page_id for page_id, _ in page_similarities[:args.top_k]]

            results.append({
                "question_id": question_id,
                "retrieved_passages": top_k_passages
            })

    return results


# Extract visual features for each image in the dataset
def get_all_image_tensors(model, dataset):
    all_image_features = {}
    eval_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dc_image,
        shuffle=False,
        drop_last=False,
    )

    for batch in tqdm(eval_dataloader):
        images, image_langs, doc_names, page_numbers = batch
        images = list(images)

        with torch.cuda.amp.autocast():
            image_fts = model.get_fused_embeddings(images=images, text=image_langs)

        for i, doc_name in enumerate(doc_names):
            page_number = page_numbers[i]
            image_feature = image_fts[i]

            if doc_name not in all_image_features:
                all_image_features[doc_name] = {}

            all_image_features[doc_name][page_number] = image_feature

    return all_image_features


# Initialize the model
model.base = model.base.half().eval().to('cuda')

# Load datasets
query_dataset = MMDocIR_dataset(data_path=args.data_path, preprocessor=None, mode='query')
image_dataset = MMDocIR_dataset(data_path=args.data_path, preprocessor=None, mode='image')

# Get image features
img_fts = get_all_image_tensors(model=model, dataset=image_dataset)

# Perform evaluation
result = eval(model=model, eval_dataset=query_dataset, all_pages_features=img_fts)

# Save the results
with open('docresult.json', 'w') as jf:
    json.dump(result, jf, indent=4)
