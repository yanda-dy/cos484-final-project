import json
import os
import random
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

MODEL_NAME = "gpt-4.1-mini"
K = 16
SEEDS = [15]
client = OpenAI(api_key="")

def format_choices(choices):
    return "\n".join([f"{c['label']}: {c['text']}" for c in choices])

def build_icl_prompt(icl_examples, test_input, mode):
    def format_example(row):
        choices_text = format_choices(row["choices"])
        correct = row["true_answer"]
        if mode == "questions_only":
            return f"Question: {row['question']}\n{choices_text}"
        elif mode == "answers_only":
            return f"Answer:\n{correct['label']}: {correct['text']}"
        else:
            raise ValueError(f"Unknown ICL mode: {mode}")

    examples = [format_example(row) for _, row in icl_examples.iterrows()]
    test_question = f"Question: {test_input['question']}\n{format_choices(test_input['choices'])}"

    return "\n\n".join(examples) + f"\n\nNow, answer the following multiple-choice question. Output only the line corresponding to your selection. Your response should be in the format ‘[letter choice]: [choice text]’.\n{test_question}\nAnswer:\n"

def build_zero_shot_prompt(test_input):
    return (
        "Now, answer the following multiple-choice question. Output only the line corresponding to your selection. "
        "Your response should be in the format ‘[letter choice]: [choice text]’.\n"
        f"Question: {test_input['question']}\n{format_choices(test_input['choices'])}\nAnswer:\n"
    )

def query_openai(prompt: str):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"

def run_experiment(data_path, seed, dataset_name):
    data = pd.read_json(data_path, lines=True).head(1000)
    assert len(data) >= K + 4

    print(f"\n=== Running experiments on dataset {dataset_name} with seed={seed} ===")

    test_set = data.copy().reset_index(drop=True)
    test_set["id"] = test_set.index.astype(str)
    data["id"] = test_set["id"]

    output_file = f"icl_question_vs_answer_{dataset_name}_{MODEL_NAME}_k={K}.jsonl"
    with open(output_file, "w") as out_f:
        for i, row in tqdm(test_set.iterrows(), total=len(test_set)):
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
            icl_qonly_prompt = build_icl_prompt(icl_examples, test_input, mode="questions_only")
            icl_aonly_prompt = build_icl_prompt(icl_examples, test_input, mode="answers_only")

            record = {
                "id": row["id"],
                "question": row["question"],
                "true_answer": row["true_answer"],
                "zero_shot_output": query_openai(zero_shot_prompt),
                "icl_questions_only_output": query_openai(icl_qonly_prompt),
                "icl_answers_only_output": query_openai(icl_aonly_prompt),
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    print(f"Done! Results saved to: {output_file}")

datasets = {
    # "CommonsenseQA": "processed_CommonsenseQA.jsonl",
    # "balanced_copa": "processed_balanced_copa.jsonl",
    "openbookqa": "processed_openbookqa.jsonl",
}

for dataset_name, data_path in datasets.items():
    for seed in SEEDS:
        run_experiment(data_path, seed, dataset_name)
