import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Merge and re-rank results based on dino similarities")
    parser.add_argument('--dino_file', type=str, default="result_dino.json", help="Path to the Dino result file")
    parser.add_argument('--merge_file', type=str, default="hebing_m2kr.json", help="Path to the merged result file")
    parser.add_argument('--query_file', type=str, default="query_data.json", help="Path to the query metadata")
    parser.add_argument('--output_file', type=str, default="result_merged_hebing.json", help="Output file path")
    return parser.parse_args()

def main(args):
    # Load input files
    with open(args.dino_file) as jf:
        data = json.load(jf)
    with open(args.merge_file) as jf:
        data2 = json.load(jf)
    with open(args.query_file) as jf:
        q = json.load(jf)

    for i, item in enumerate(data):
        question_id = item['question_id']
        # Determine the valid prefix and similarity threshold
        if 0 <= i <= 1399:
            valid_prefix = "A"
            max_sim = 0
        elif 1400 <= i <= 4414:
            valid_prefix = "Q"
            max_sim = 0.96
        else:
            valid_prefix = "B"
            max_sim = 0.88

        if item['max_similarity'] > max_sim:
            item_passage = item['retrieved_passages'][0]

            if valid_prefix == "A":
                continue

            if item_passage.startswith(valid_prefix):
                print(i)
                first_five_passages = data2[i]['retrieved_passages'][:5]

                if item_passage in first_five_passages:
                    data2[i]['retrieved_passages'].remove(item_passage)
                    data2[i]['retrieved_passages'] = [item_passage] + data2[i]['retrieved_passages']
                else:
                    data2[i]['retrieved_passages'] = [item_passage] + data2[i]['retrieved_passages']

    # Save to output file
    with open(args.output_file, 'w') as jf:
        json.dump(data2, jf, indent=4)
    print(f"Saved updated results to {args.output_file}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
