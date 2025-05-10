import json
import re
from collections import defaultdict
from math_dapo import *

def parse_answer(answer):
    try:
        return normalize_final_answer(remove_boxed(last_boxed_only_string(answer)))
    except Exception as e:
        return None

if __name__ == "__main__":
    with open("output/MATH/MATH_gpt-3.5-turbo_k4.json", 'r') as f:
        results = json.load(f)
    
    with open("data/MATH/MATH.json", 'r') as f:
        dataset = json.load(f)

    stats = defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0})

    errors = 0
    for item in results:
        problem = dataset[item["idx"]]
        if problem["difficulty"] not in "3":
            continue



        exp_name = item["experiment_name"]
        stats[exp_name]["total"] += 1
        resp = parse_answer(item["response"])
        gt = item["answer"]
        # print(resp, gt)
        if resp == gt:
            stats[exp_name]["correct"] += 1
        if resp is None:
            errors += 1
            stats[exp_name]["errors"] += 1
    
    print(f"Errors: {errors}")
    for exp_name, stat in stats.items():
        print(f"{exp_name}: {stat['correct']}/{stat['total']} ({stat['correct']/stat['total']:.2%}) (Errors: {stat['errors']})")
    