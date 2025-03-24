import json
import os
import argparse
from collections import defaultdict


# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Merge and rank retrieved passages")

    # Add arguments for file paths and hyperparameters
    parser.add_argument('--path1', type=str, required=True, help='Path to the first input file')
    parser.add_argument('--path2', type=str, required=True, help='Path to the second input file')
    parser.add_argument('--path3', type=str, required=True, help='Path to the third input file')
    parser.add_argument('--path4', type=str, required=True, help='Path to the fourth input file')
    parser.add_argument('--path5', type=str, required=True, help='Path to the fifth input file')
    parser.add_argument('--data_dino', type=str, required=True, help='Path to the data_dino file')
    parser.add_argument('--similarity_threshold', type=float, default=0.35, help='Threshold for similarity validation')

    return parser.parse_args()


# Function to check if the similarity passes the threshold
def panding(data_dino, data2, similarity_threshold=0.35):
    k = 0
    p = 0
    for i, item in enumerate(data_dino):
        question_id = item['question_id']

        # Determine the valid prefix based on question_id
        if 0 <= i <= 1399:
            valid_prefix = "A"
            max_similarity = 0
        elif 1400 <= i <= 4414:
            valid_prefix = "Q"
            max_similarity = 0.95
        else:
            valid_prefix = "B"
            max_similarity = 0.88

        if item['max_similarity'] > max_similarity:
            item_passage = item['retrieved_passages'][0]
            if valid_prefix == "A":
                continue
            k += 1

            # Check if the item_passage has the correct prefix
            if item_passage.startswith(valid_prefix):
                print(i)
                first_five_passages = data2[i]['retrieved_passages'][:5]

                # Check if the top item_passage is in the first 5 in data2
                if item_passage in first_five_passages:
                    p += 1
                    data2[i]['retrieved_passages'].remove(item_passage)
                    data2[i]['retrieved_passages'] = [item_passage] + data2[i]['retrieved_passages']
                else:
                    data2[i]['retrieved_passages'] = [item_passage] + data2[i]['retrieved_passages']
    return p / k > similarity_threshold


# Function to merge rankings from multiple systems
def merge_rankings(input_data):
    question_groups = defaultdict(list)
    for item in input_data:
        question_groups[item["question_id"]].append(item["retrieved_passages"])

    result = []

    for q_id, sys_lists in question_groups.items():
        all_passages = set()
        for sys in sys_lists:
            all_passages.update(sys)
        all_passages = list(all_passages)

        avg_ranks = {}
        for pid in all_passages:
            total = 0.0
            for sys in sys_lists:
                try:
                    rank = sys.index(pid) + 1
                except ValueError:
                    rank = 22
                total += rank
            avg_ranks[pid] = total / len(sys_lists)

        sorted_passages = sorted(all_passages, key=lambda x: (avg_ranks[x], x))

        result.append({
            "question_id": q_id,
            "retrieved_passages": sorted_passages
        })

    return result


# Main function to load data and perform merging and ranking
def tes_merge_rankings_m2kr(args):
    input_data = []

    # Load data from each of the input paths and apply the panding function
    with open(args.path1) as jf:
        data1 = json.load(jf)
    if panding(args.data_dino, data1, args.similarity_threshold):
        input_data.extend(data1)

    with open(args.path2) as jf:
        data2 = json.load(jf)
    if panding(args.data_dino, data2, args.similarity_threshold):
        input_data.extend(data2)

    with open(args.path3) as jf:
        data3 = json.load(jf)
    if panding(args.data_dino, data3, args.similarity_threshold):
        input_data.extend(data3)

    with open(args.path4) as jf:
        data4 = json.load(jf)
    if panding(args.data_dino, data4, args.similarity_threshold):
        input_data.extend(data4)

    with open(args.path5) as jf:
        data5 = json.load(jf)
    if panding(args.data_dino, data5, args.similarity_threshold):
        input_data.extend(data5)

    if len(input_data) == 0:
        input_data = data1

    # Merge rankings
    result = merge_rankings(input_data)

    # Save the results to a file
    with open("hebing_m2kr.json", 'w') as jf:
        json.dump(result, jf, indent=4)

    # Print the result
    print(result)

def tes_merge_rankings_doc(args):
    input_data = []
    # Load data from each of the input paths and apply the panding function
    with open(args.path1) as jf:
        data1 = json.load(jf)
    with open(args.path2) as jf:
        data2 = json.load(jf)
        input_data.extend(data2)
    with open(args.path3) as jf:
        data3 = json.load(jf)
        input_data.extend(data3)
    with open(args.path4) as jf:
        data4 = json.load(jf)
        input_data.extend(data4)
    with open(args.path5) as jf:
        data5 = json.load(jf)
        input_data.extend(data5)
    if len(input_data) == 0:
        input_data = data2
    # Merge rankings
    result = merge_rankings(input_data)

    # Save the results to a file
    with open("hebing_doc.json", 'w') as jf:
        json.dump(result, jf, indent=4)
    # Print the result
    print(result)


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    # Call the merging function with the parsed arguments
    if args.mode == "m2kr":
        tes_merge_rankings_m2kr(args)
    else:
        tes_merge_rankings_doc(args)
"""
python merge_rank.py \
  --path1 to/your_path/file1.json \
  --path2 to/your_path/file2.json \
  --path3 to/your_path/file3.json \
  --path4 to/your_path/file4.json \
  --path5 to/your_path/file5.json \
  --data_dino to/your_path/data_dino.json \
  --similarity_threshold 0.35 \
  --mode m2kr
"""

"""
python merge_rank.py \
  --path1 to/your_path/file1.json \
  --path2 to/your_path/file2.json \
  --path3 to/your_path/file3.json \
  --path4 to/your_path/file4.json \
  --path5 to/your_path/file5.json \
  --data_dino to/your_path/data_dino.json \
  --similarity_threshold 0.35 \
  --mode doc

"""