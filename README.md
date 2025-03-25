# MDR

Using-GPU：A100-40G*8

Used Training data: Link：https://pan.baidu.com/s/1tF8CSWhVbwevfAPaipvRYQ?pwd=7nui  password：7nui 

Run run.sh to run our step-by step code, You shoule replace the data path to  your own path. 
you can download GME-7B model and dino-v2 as backbone.
the pipeline is :
1. first using cv_tools dino to recognize the visual keypoints, here you get a visual-keypoints results.
2. train five different parameters expert to vote for both two tasks, the five experts total is a unified model.
3. merge the experts' voting results and  the visual keypoints results to get
The five experts and the dino as a whole can be viewed as a unified model. They are useful for both tasks and do not need to be trained specifically for the task.
![image](https://github.com/user-attachments/assets/3ba34e31-bf8a-4a6d-a3fc-b22848a922f3)

You can download Several dataset by M2KR-train-dataset. You need to transfer the json file to CSV file and merge the two results so that you can submit it to the leaderboard. 

You can run a single Expert as a simple version of our system by run run2.sh

| **变量名 / 路径**                | **当前值**                                     | **说明**                       | **需要替换为**                 |
| -------------------------------- | ---------------------------------------------- | ------------------------------ | ------------------------------ |
| `GME_PATH`                       | `to/your_path/gme-Qwen2-VL-7B-Instruct`        | 主模型路径                     | 模型所在实际路径               |
| `M2KR_DATA_PATH`                 | `to/your_path/images`                          | M2KR 图像数据目录              | 图像数据实际路径               |
| `M2KR_ChALLENGE_PATH`            | `to/your_path/MMIR/M2KR/Challenge`             | M2KR 挑战图像原始路径          | 替换成图像 challenge 路径      |
| `DOC_DATA_PATH`                  | `to/your_path/page_images`                     | 文档图像目录                   | 文档页面图像路径               |
| `DOC_PASSAGE`                    | `to/your_path/doc_passage.json`                | 文档段落 JSON 文件             | 实际段落数据路径               |
| `DOC_QUERY`                      | `to/your/doc_query.json`                       | 文档查询 JSON 文件             | 实际查询数据路径               |
| `CLEAN_DATA_PATH`                | `to/your_path/clean_queries.json`              | 过滤后的 clean query 数据路径  | 实际 clean query 路径          |
| `M2KR_PASSAGES_PATH`             | `to/your_path/passages.json`                   | M2KR 段落数据路径              | 实际段落 JSON 路径             |
| `SPLIT_IMAGE_PATH`               | `to/your_path/split_images`                    | 分割后的图像保存路径           | 图像切割结果保存路径           |
| `ARXIV_PAGES`                    | `to/your_path/arxiv_pages`                     | arXiv 图像页面路径             | 替换为实际 arXiv 图像目录      |
| `M2KR_QUERY`                     | `to/your_path/m2kr_query_data_path`            | M2KR 查询数据路径              | 查询数据 JSON 路径             |
| `dino_model_path`                | `to/your_path/dinov2-large`                    | DINOv2 模型路径                | 预训练视觉模型路径             |
| `generated_data`                 | `to/your_path/generated_queries.json`          | 自动生成的 query 数据路径      | 实际路径                       |
| `info_seek_path`                 | `to/your_path/infoseek_passages.json`          | InfoSeek 段落数据              | 替换成实际路径                 |
| `infoseek_data_path`             | `to/your_path/infoseek_used_images2`           | InfoSeek 图像路径              | 替换成实际路径                 |
| `tat_pages`                      | `to/your_path/tat_pages`                       | TAT 页数据路径                 | 替换成实际路径                 |
| `output_dir`（多个地方）         | `"gme_exp1"` ~ `"gme_exp5"`                    | 每轮训练结果保存路径           | 可保留，或替换为你的输出目录名 |
| `save_dir`（多个地方）           | `to/your_path/m2kr_resultX.json / doc_resultX` | 模型推理输出文件               | 替换为你希望的保存路径         |
| `path1~5` in `merge_rank.py`     | `to/your_path/m2kr_resultX.json / doc_resultX` | 多轮推理结果 JSON 文件输入路径 | 替换为实际生成的路径           |
| `data_dino` in `merge_rank.py`   | `to/your_path/data_dino.json`                  | DINO similarity 数据路径       | 实际 dino 相似度 JSON 路径     |
| `dino_file` in `merge_result.py` | `to/your_path/result_dino.json`                | DINO 最终推理文件              | 实际推理路径                   |
| `merge_file`                     | `to/your_path/hebing_m2kr.json`                | 合并后的多模型结果路径         | 替换为你要合并保存的路径       |
| `output_file`                    | `to/your_path/result_merged_hebing.json`       | 最终融合输出结果路径           | 替换为最终输出保存路径         |

------
