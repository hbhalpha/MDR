#!/bin/bash

# 设置通用参数
GME_PATH="to/your_path/gme-Qwen2-VL-7B-Instruct"
M2KR_DATA_PATH="to/your_path/images"
M2KR_ChALLENGE_PATH = "to/your_path/MMIR/M2KR/Challenge"
DOC_DATA_PATH="to/your_path/page_images"
DOC_PASSAGE="to/your_path/doc_passage.json"
DOC_QUERY="to/your/doc_query.json"
CLEAN_DATA_PATH="to/your_path/clean_queries.json"
M2KR_PASSAGES_PATH="to/your_path/passages.json"
SPLIT_IMAGE_PATH="to/your_path/split_images"
ARXIV_PAGES = "to/your_path/arxiv_pages"
M2KR_QUERY = "to/your_path/m2kr_query_data_path"

python ./cv_test/clean_pic.py \
  --input_dir  "$M2KR_ChALLENGE_PATH"\
  --output_dir  "$SPLIT_IMAGE_PATH" \
  --ext .png \
  --min_area 5000 \
  --max_splits 10

python ./cv_test/dinov24test.py \
  --query_data_path "$M2KR_QUERY" \
  --passages_path "$M2KR_PASSAGES_PATH" \
  --m2kr_image_path "$M2KR_DATA_PATH" \
  --split_image_path "$SPLIT_IMAGE_PATH" \
  --dino_model_path to/your_path/dinov2-large \
  --output_path to/your_path/result_dino.json

python ./fine-tune/ft.py \
  --gme_path to/your_path/gme-Qwen2-VL-7B-Instruct \
  --m2kr_train_data_path to/your_path/m2kr_images \
  --doc_data_path to/your_path/page_images \
  --clean_data_path to/your_path/clean_queries.json \
  --info_seek_path to/your_path/infoseek_passages.json \
  --infoseek_data_path to/your_path/infoseek_used_images2 \
  --generated_data to/your_path/generated_queries.json \
  --trained_passages_path to/your_path/m2kr_passages.json \
  --arxiv_pages to/your_path/arxiv_pages \
  --tat_pages to/your_path/tat_pages \
  --output_dir "gme_exp1" \
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --learning_rate 2e-5 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1

# 第一轮训练：学习率 5e-5
python ./zhengliu/train.py \
  --gme_path "$GME_PATH" \
  --m2kr_data_path "$M2KR_DATA_PATH" \
  --doc_data_path "$DOC_DATA_PATH" \
  --clean_data_path "$CLEAN_DATA_PATH" \
  --M2KR_passages_path "$M2KR_PASSAGES_PATH" \
  --output_dir "gme_exp2" \
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --learning_rate 5e-5 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1

# 第二轮训练：学习率 1e-5
python ./zhengliu/train.py \
  --gme_path "$GME_PATH" \
  --m2kr_data_path "$M2KR_DATA_PATH" \
  --doc_data_path "$DOC_DATA_PATH" \
  --clean_data_path "$CLEAN_DATA_PATH" \
  --M2KR_passages_path "$M2KR_PASSAGES_PATH" \
  --output_dir "gme_exp3" \
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1
# 第二轮训练：学习率 1e-5
python ./zhengliu/train.py \
  --gme_path "$GME_PATH" \
  --m2kr_data_path "$M2KR_DATA_PATH" \
  --doc_data_path "$DOC_DATA_PATH" \
  --clean_data_path "$CLEAN_DATA_PATH" \
  --M2KR_passages_path "$M2KR_PASSAGES_PATH" \
  --output_dir "gme_exp4" \
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 2
# 第二轮训练：学习率 1e-5
python ./zhengliu/train.py \
  --gme_path "$GME_PATH" \
  --m2kr_data_path "$M2KR_DATA_PATH" \
  --doc_data_path "$DOC_DATA_PATH" \
  --clean_data_path "$CLEAN_DATA_PATH" \
  --M2KR_passages_path "$M2KR_PASSAGES_PATH" \
  --output_dir "gme_exp5" \
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 2
python ./gme_m2krlora.py  --gme_path "$GME_PATH" --peft_model_path "gme_exp1" --dataset_path "$M2KR_DATA_PATH" --query_path "$M2KR_QUERY" --passage_path "$M2KR_PASSAGES_PATH" --save_dir "to/your_path/m2kr_reuslt1.json"
python ./gme_m2krlora.py  --gme_path "$GME_PATH" --peft_model_path "gme_exp2" --dataset_path "$M2KR_DATA_PATH" --query_path "$M2KR_QUERY" --passage_path "$M2KR_PASSAGES_PATH" --save_dir "to/your_path/m2kr_reuslt2.json"
python ./gme_m2krlora.py  --gme_path "$GME_PATH" --peft_model_path "gme_exp3" --dataset_path "$M2KR_DATA_PATH" --query_path "$M2KR_QUERY" --passage_path "$M2KR_PASSAGES_PATH" --save_dir "to/your_path/m2kr_reuslt3.json"
python ./gme_m2krlora.py  --gme_path "$GME_PATH" --peft_model_path "gme_exp4" --dataset_path "$M2KR_DATA_PATH" --query_path "$M2KR_QUERY" --passage_path "$M2KR_PASSAGES_PATH" --save_dir "to/your_path/m2kr_reuslt4.json"
python ./gme_m2krlora.py  --gme_path "$GME_PATH" --peft_model_path "gme_exp5" --dataset_path "$M2KR_DATA_PATH" --query_path "$M2KR_QUERY" --passage_path "$M2KR_PASSAGES_PATH" --save_dir "to/your_path/m2kr_reuslt5.json"
python script.py --gme_path "$GME_PATH" --peft_model_path "gme_exp1" --data_path "$DOC_DATA_PATH" --passages_file "$DOC_PASSAGE" --queries_file "$DOC_QUERY" --save_dir "to/your_path/doc_reuslt1.json"
python script.py --gme_path "$GME_PATH" --peft_model_path "gme_exp2" --data_path "$DOC_DATA_PATH" --passages_file "$DOC_PASSAGE" --queries_file "$DOC_QUERY" --save_dir "to/your_path/doc_reuslt2.json"
python script.py --gme_path "$GME_PATH" --peft_model_path "gme_exp3" --data_path "$DOC_DATA_PATH" --passages_file "$DOC_PASSAGE" --queries_file "$DOC_QUERY" --save_dir "to/your_path/doc_reuslt3.json"
python script.py --gme_path "$GME_PATH" --peft_model_path "gme_exp4" --data_path "$DOC_DATA_PATH" --passages_file "$DOC_PASSAGE" --queries_file "$DOC_QUERY" --save_dir "to/your_path/doc_reuslt4.json"
python script.py --gme_path "$GME_PATH" --peft_model_path "gme_exp5" --data_path "$DOC_DATA_PATH" --passages_file "$DOC_PASSAGE" --queries_file "$DOC_QUERY" --save_dir "to/your_path/doc_reuslt5.json"


python merge_rank.py \
  --path1 to/your_path/m2kr_reuslt1.json \
  --path2 to/your_path/m2kr_reuslt2.json \
  --path3 to/your_path/m2kr_reuslt3.json \
  --path4 to/your_path/m2kr_reuslt4.json \
  --path5 to/your_path/m2kr_reuslt5.json \
  --data_dino to/your_path/data_dino.json \
  --similarity_threshold 0.35 \
  --mode m2kr

python merge_rank.py \
  --path1 to/your_path/doc_reuslt1.json \
  --path2 to/your_path/doc_reuslt2.json \
  --path3 to/your_path/doc_reuslt3.json \
  --path4 to/your_path/doc_reuslt4.json \
  --path5 to/your_path/doc_reuslt5.json \
  --data_dino to/your_path/data_dino.json \
  --similarity_threshold 0.35 \
  --mode doc


python merge_result.py \
  --dino_file to/your_path/result_dino.json \
  --merge_file to/your_path/hebing_m2kr.json \
  --query_file "$M2KR_QUERY" \
  --output_file to/your_path/result_merged_hebing.json
