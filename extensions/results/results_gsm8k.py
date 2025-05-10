import json
import re
from collections import defaultdict

def parse_answer(answer):
    answer = answer.replace("=", " ")
    answer = answer.replace(",", "")
    answer_norm = re.sub(r"[^0-9 ]", " ", answer)
    answer_norm = answer_norm.strip().split()
    for item in answer_norm[::-1]:
        try:
            x = int(item)
            return x
        except:
            pass

    return None

if __name__ == "__main__":
    with open("output/GSM8K/gsm8k_test_qwen2.5-0.5b-instruct_k4.json", 'r') as f:
        results = json.load(f)
    
    with open("output/GSM8k/gsm8k_test_qwen2.5-0.5b-instruct_k4_mis.json", 'r') as f:
        results.extend(json.load(f))
    
    with open("data/GSM8K/gsm8k_test.json", 'r') as f:
        dataset = json.load(f)
    
    stats = defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0})

    errors = 0
    for item in results:
        exp_name = item["experiment_name"]
        stats[exp_name]["total"] += 1
        resp = parse_answer(item["response"])
        if resp == parse_answer(item["answer"]):
            stats[exp_name]["correct"] += 1
        # else:
        #     print(parse_answer(item["response"]), parse_answer(item["answer"]))
        #     print(f"Error in response: {item['idx']} {item['experiment_name']} {item['response']}")
        if resp is None:
            errors += 1
            stats[exp_name]["errors"] += 1
            # print(f"Error in response: {item['idx']} {item['experiment_name']} {item['response']}")
    
    print(f"Errors: {errors}")
    for exp_name, stat in stats.items():
        print(f"{exp_name}: {stat['correct']}/{stat['total']} ({stat['correct']/stat['total']:.2%})")
    