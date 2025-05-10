import json
import os
import random
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# config
MODEL_NAME = "gpt-4.1-mini"
K = 16
SEEDS = [15]
client = OpenAI(api_key="")

def build_prompt_mrpc(examples, test_row, use_random_labels=False):
    def format_example(i, row, use_random=False):
        label = random.choice(["equivalent", "not_equivalent"]) if use_random else row["true_label"]
        return f"Example {i}\nsentence1: \"{row['sentence1']}\" sentence2: \"{row['sentence2']}\"\nLabel: {label}"

    prompt_parts = [
        format_example(i + 1, row, use_random=use_random_labels)
        for i, (_, row) in enumerate(examples.iterrows())
    ]

    test_prompt = (
        "Now, given the following two sentences, classify their relationship as either semantically "
        "\"equivalent\" or \"not_equivalent\". Respond with only one of the following labels: \"equivalent\" or \"not_equivalent\".\n"
        f"sentence1: \"{test_row['sentence1']}\", sentence2: \"{test_row['sentence2']}\"\nLabel:"
    )
    return "\n\n".join(prompt_parts) + "\n\n" + test_prompt

def build_prompt_rte(examples, test_row, use_random_labels=False):
    def format_example(i, row, use_random=False):
        label = random.choice(["entailment", "not_entailment"]) if use_random else row["true_label"]
        return f"Example {i}\nsentence1: \"{row['sentence1']}\" , sentence2: \"{row['sentence2']}\"\nLabel: {label}"

    prompt_parts = [
        format_example(i + 1, row, use_random=use_random_labels)
        for i, (_, row) in enumerate(examples.iterrows())
    ]

    test_prompt = (
        "\nNow, given the following two sentences, classify whether sentence2 is logically "
        "\"entailment\" or \"not_entailment\" by sentence1. Respond with only one of the following labels: \"entailment\" or \"not_entailment\".\n"
        f"sentence1: \"{test_row['sentence1']}\",  sentence2: \"{test_row['sentence2']}\"\nLabel:"
    )
    return "\n\n".join(prompt_parts) + "\n\n" + test_prompt

def build_prompt_tweeteval_hate(examples, test_row, use_random_labels=False):
    def format_example(i, row, use_random=False):
        label = random.choice(["hate", "non_hate"]) if use_random else row["true_label"]
        return f"Example {i}\nsentence1: \"{row['sentence1']}\"\nLabel: {label}"

    prompt_parts = [
        format_example(i + 1, row, use_random=use_random_labels)
        for i, (_, row) in enumerate(examples.iterrows())
    ]

    test_prompt = (
        "\nNow, classify the sentiment of the following sentence as either \"hate\" or \"non_hate\". Respond with only one of the following labels: \"hate\" or \"non_hate\".\n"
        f"sentence1: \"{test_row['sentence1']}\"\nLabel:"
    )
    return "\n\n".join(prompt_parts) + "\n\n" + test_prompt

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

PROMPT_BUILDERS = {
    "mrpc": build_prompt_mrpc,
    "rte": build_prompt_rte,
    "tweeteval_hate": build_prompt_tweeteval_hate,
}

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
    assert len(data) >= K + 4, f"{dataset_name} must have at least K + 4 examples"

    test_set = data.copy().reset_index(drop=True)
    prompt_fn = PROMPT_BUILDERS[dataset_name]
    output_path = f"baseline_{dataset_name}_{MODEL_NAME}_k={K}.jsonl"

    print(f"\n=== Running {dataset_name} | seed={seed} ===")

    with open(output_path, "w") as out_f:
        for i, row in tqdm(test_set.iterrows(), total=len(test_set)):
            per_example_seed = hash((seed, row["id"])) % (2**32)
            random.seed(per_example_seed)

            candidate_pool = data[data["id"] != row["id"]]
            icl_examples = candidate_pool.sample(n=K, random_state=per_example_seed)

            zero_shot_prompt = prompt_fn(pd.DataFrame(), row, use_random_labels=False)
            icl_random_prompt = prompt_fn(icl_examples, row, use_random_labels=True)
            icl_gold_prompt = prompt_fn(icl_examples, row, use_random_labels=False)

            # === DEBUG: Inspect prompt formatting ===
            # print(f"\n--- Example #{i} | ID: {row['id']} ---")
            # print("\n[Zero-Shot Prompt]:\n", zero_shot_prompt)
            # print("\n[ICL Prompt (Random Labels)]:\n", icl_random_prompt)
            # print("\n[ICL Prompt (Gold Labels)]:\n", icl_gold_prompt)
            # print("\n--- End of Prompts ---\n")

            record = {
                "id": row["id"],
                "true_label": row["true_label"],
                "zero_shot_output": query_openai(zero_shot_prompt),
                "icl_random_output": query_openai(icl_random_prompt),
                "icl_gold_output": query_openai(icl_gold_prompt),
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    print(f"✅ Results saved to {output_path}")

DATASETS = {
    # "mrpc": "processed_mrpc.jsonl",
    "rte": "processed_rte.jsonl",
    # "tweeteval_hate": "processed_tweeteval_hate.jsonl",
}

for name, path in DATASETS.items():
    for seed in SEEDS:
        run_experiment(path, seed, name)