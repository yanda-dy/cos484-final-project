import json
import os
import random
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

MODEL_NAME = "gpt-4.1-mini"
K = 32
SEEDS = [15]
client = OpenAI(api_key="")

def build_prompt(icl_examples, test_input, use_random_labels=False):
    def format_choices(choices):
        return "\n".join([f"{c['label']}: {c['text']}" for c in choices])

    def format_example(row, use_random=False):
        choices_text = format_choices(row["choices"])
        if use_random:
            random_choice = random.choice(row["choices"])
            answer_line = f"{random_choice['label']}: {random_choice['text']}"
        else:
            correct = row["true_answer"]
            answer_line = f"{correct['label']}: {correct['text']}"
        return f"Question: {row['question']}\n{choices_text}\nAnswer:\n{answer_line}"

    examples = [format_example(row, use_random=use_random_labels) for _, row in icl_examples.iterrows()]
    
    # Format test question without the answer
    test_question = f"Question: {test_input['question']}\n" + format_choices(test_input["choices"])

    return "\n\n".join(examples) + f"\n\nNow, answer the following multiple-choice question. Output only the line corresponding to your selection. Your response should be in the format ‘[letter choice]: [choice text]’.\n{test_question}\nAnswer:\n"

def build_zero_shot_prompt(test_input):
    def format_choices(choices):
        return "\n".join([f"{c['label']}: {c['text']}" for c in choices])

    question_text = f"Question: {test_input['question']}"
    choices_text = format_choices(test_input["choices"])

    return (
        "Now, answer the following multiple-choice question. Output only the line corresponding to your selection. "
        "Your response should be in the format ‘[letter choice]: [choice text]’.\n"
        f"{question_text}\n{choices_text}\nAnswer:\n"
    )

def query_openai(prompt: str):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"

def run_experiment(data_path, seed, dataset_name):
    data = pd.read_json(data_path, lines=True).head(1000)
    assert len(data) >= K + 4, f"Dataset {dataset_name} must have at least K + 4 examples"

    print(f"\n=== Running experiments on dataset {dataset_name} with seed={seed} ===")

    test_set = data.copy().reset_index(drop=True)
    if "ID" in test_set.columns:
        test_set["id"] = test_set["ID"].astype(str)
    else:
        test_set["id"] = test_set.index.astype(str)
    data["id"] = test_set["id"] 

    output_file = f"baseline_{dataset_name}_{MODEL_NAME}_k={K}.jsonl"
    with open(output_file, "w") as out_f:
        for i, row in tqdm(test_set.iterrows(), total=len(test_set)):
            # === Per-example ICL sampling ===
            per_example_seed = hash((seed, row["id"])) % (2**32)
            random.seed(per_example_seed)

            candidate_pool = data[data["id"] != row["id"]]
            icl_examples = candidate_pool.sample(n=K, random_state=per_example_seed)

            test_input = {
                "id": row["id"],
                "question": row["question"],
                "choices": row["choices"],
                "true_answer": row["true_answer"]
            }

            zero_shot_prompt = build_zero_shot_prompt(test_input)
            icl_random_prompt = build_prompt(icl_examples, test_input, use_random_labels=True)
            icl_gold_prompt = build_prompt(icl_examples, test_input, use_random_labels=False)
            
            record = {
                "id": row["id"],
                "question": row["question"],
                "true_answer": row["true_answer"],
                "zero_shot_output": query_openai(zero_shot_prompt),
                "icl_random_output": query_openai(icl_random_prompt),
                "icl_gold_output": query_openai(icl_gold_prompt),
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    print(f"Results saved to: {output_file}")

datasets = {
    "CommonsenseQA": "processed_CommonsenseQA.jsonl",
    # "balanced_copa": "processed_balanced_copa.jsonl",
    # "openbookqa": "processed_openbookqa.jsonl",
}

for dataset_name, data_path in datasets.items():
    for seed in SEEDS:
        run_experiment(data_path, seed, dataset_name)
