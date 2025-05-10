import json
import re
from collections import defaultdict

def parse_answer(answer, question):
    if answer.startswith("ERROR"):
        return None

    if answer.startswith("Answer: "):
        answer = answer[8:]

    answer_norm = re.sub(r"[^a-zA-Z0-9]", " ", answer)
    answer_norm = answer_norm.strip().split()
    for item in answer_norm:
        if item in list("12345"):
            return chr(int(item) - 1 + ord('A'))
        if item in list("ABCDE"):
            return item
    
    return None

if __name__ == "__main__":
    for k in [1, 2, 4, 8, 16, 32]:
        print(f"k: {k}")

        model = "qwen2.5-0.5b-instruct"
        with open(f"output/CSQA/csqa_train_{model}_k{k}_l.json", 'r') as f:
            results = json.load(f)
        with open(f"output/CSQA/csqa_train_{model}_k{k}_rd.json", 'r') as f:
            results.extend(json.load(f))

        model = "smollm2-360m-instruct"
        # model = "llama-3.2-1b-instruct"
        with open(f"output/CSQA/csqa_train_{model}_k{k}_full.json", 'r') as f:
            results = json.load(f)
        
        with open("data/CSQA/csqa_train.json", 'r') as f:
            dataset = json.load(f)
        
        stats = defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0})

        errors = 0
        for item in results:
            exp_name = item["experiment_name"]
            stats[exp_name]["total"] += 1
            resp = parse_answer(item["response"], dataset[item["idx"]])
            if resp == item["answerKey"]:
                stats[exp_name]["correct"] += 1
            if resp is None:
                errors += 1
                stats[exp_name]["errors"] += 1
                # print(f"Error in response: {item['idx']} {item['experiment_name']} {item['response']}")
        
        print(f"Errors: {errors}")
        for exp_name, stat in stats.items():
            print(f"{exp_name}: {stat['correct']}/{stat['total']} ({stat['correct']/stat['total']:.2%})")
        
        print()