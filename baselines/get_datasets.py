import json
import pandas as pd
from datasets import load_dataset

def get_commonsenseqa(filepath, save_path):
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            item = json.loads(line)
            q = item["question"]
            stem = q["stem"]
            choices = q["choices"]
            answer_label = item["answerKey"]

            # fetch correct answer
            true_answer = next(c for c in choices if c["label"] == answer_label)

            rows.append({
                "id": item["id"],
                "question": stem,
                "true_answer": true_answer,
                "choices": choices
            })

    df = pd.DataFrame(rows)
    df.to_json(save_path, orient="records", lines=True)
    print(f"✅ Processed CommonSenseQA saved to: {save_path}")
    return df

def get_balancedcopa(save_path, split="train"):
    splits = {'train': 'train.csv', 'test': 'test.csv'}
    hf_path = f"hf://datasets/pkavumba/balanced-copa/{splits[split]}"
    
    df = pd.read_csv(hf_path)
    processed = []

    for _, row in df.iterrows():
        if row["question"].strip().lower() == "cause":
            question_text = f"What was the cause of this? {row['premise'].strip()}"
        else:
            question_text = f"What happened as a result? {row['premise'].strip()}"

        correct_label = "A" if row["label"] == 0 else "B"
        correct_text = row["choice1"] if correct_label == "A" else row["choice2"]

        choices = [
            {"label": "A", "text": row["choice1"].strip()},
            {"label": "B", "text": row["choice2"].strip()}
        ]

        entry = {
            "id": str(row["id"]),
            "question": question_text,
            "true_answer": {
                "label": correct_label,
                "text": correct_text.strip()
            },
            "choices": choices
        }

        processed.append(entry)

    # save as jsonl
    with open(save_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

def get_openbookqa(save_path, split="train"):
    dataset = load_dataset("openbookqa", "main", split=split)

    processed = []

    for row in dataset:
        choices_dict = eval(row["choices"]) if isinstance(row["choices"], str) else row["choices"]
        choices = [
            {"label": label, "text": text.strip()}
            for label, text in zip(choices_dict["label"], choices_dict["text"])
        ]

        answer_label = row["answerKey"].strip()
        correct_text = next((c["text"] for c in choices if c["label"] == answer_label), None)

        entry = {
            "id": str(row["id"]),
            "question": row["question_stem"].strip(),
            "true_answer": {
                "label": answer_label,
                "text": correct_text
            },
            "choices": choices
        }

        processed.append(entry)

    # save as jsonl
    with open(save_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

def get_ai2_arc(save_path, difficulty="ARC-Easy", split="train"):
    dataset = load_dataset("ai2_arc", difficulty, split=split)

    processed = []

    for row in dataset:
        question = row["question"]
        choices_list = row["choices"]["text"]
        labels_list = row["choices"]["label"]
        answer_label = row["answerKey"]

        choices = [
            {"label": label, "text": text.strip()}
            for label, text in zip(labels_list, choices_list)
        ]

        correct_text = next((c["text"] for c in choices if c["label"] == answer_label), None)

        entry = {
            "id": str(row["id"]),
            "question": question.strip(),
            "true_answer": {
                "label": answer_label,
                "text": correct_text
            },
            "choices": choices
        }

        processed.append(entry)

    # save as jsonl
    with open(save_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

def get_mrpc(save_path, split="train"):
    dataset = load_dataset("nyu-mll/glue", "mrpc", split=split)

    label_map = {1: "equivalent", 0: "not_equivalent"}

    processed = []

    for example in dataset:
        entry = {
            "id": str(example["idx"]),
            "sentence1": example["sentence1"].strip(),
            "sentence2": example["sentence2"].strip(),
            "true_label": label_map[int(example["label"])]
        }
        processed.append(entry)

    # save as jsonl
    with open(save_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

def get_rte(save_path, split="train"):
    dataset = load_dataset("nyu-mll/glue", "rte", split=split)

    label_map = {1: "not_entailment", 0: "entailment"}

    processed = []

    for example in dataset:
        entry = {
            "id": str(example["idx"]),
            "sentence1": example["sentence1"].strip(),
            "sentence2": example["sentence2"].strip(),
            "true_label": label_map[int(example["label"])]
        }
        processed.append(entry)

    # save as jsonl
    with open(save_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

def get_tweeteval_hate(save_path, split="train"):
    dataset = load_dataset("tweet_eval", "hate", split=split)

    label_map = {0: "non_hate", 1: "hate"}

    processed = []

    for i, example in enumerate(dataset):
        entry = {
            "id": str(i),
            "sentence1": example["text"].strip(),
            "true_label": label_map[int(example["label"])]
        }
        processed.append(entry)

    # save as jsonl
    with open(save_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

# get_commonsenseqa("CommonsenseQA.jsonl","processed_CommonsenseQA.jsonl")
# get_balancedcopa(save_path="processed_balanced_copa.jsonl", split="train")
# get_openbookqa(save_path="processed_openbookqa.jsonl", split="train")
# get_ai2_arc(save_path="processed_ai2_arc_easy.jsonl", difficulty="ARC-Easy", split="train")
# get_ai2_arc(save_path="processed_ai2_arc_challenge.jsonl", difficulty="ARC-Challenge", split="train")

# get_mrpc(save_path="processed_mrpc.jsonl", split="train")
# get_rte(save_path="processed_rte.jsonl", split="train")
# get_tweeteval_hate(save_path="processed_tweeteval_hate.jsonl", split="train")