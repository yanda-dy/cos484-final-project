import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils
from similarity import find_similar_questions

def run_experiment_math_openai(client: OpenAI, dataset, dataset_full, model, num_examples, output_file, metadata):
    history = []

    example_rng = np.random.RandomState(1)
    num_problems = len(dataset_full)
    limit = min(len(dataset), 1000)

    for shuff_idx in tqdm(range(limit)):
        i = dataset[shuff_idx]["idx"]
        icl_examples = example_rng.choice(num_problems-1, num_examples, replace=False)
        icl_examples = [x if x < i else x+1 for x in icl_examples]
        if i in icl_examples:
            raise Exception("icl_examples should not contain the current problem")
        icl_examples.append(i)
        replace_indices = []
        for j in range(num_examples):
            idx = example_rng.randint(0, num_problems-1)
            idx = idx if idx < i else idx + 1
            replace_indices.append(idx)

        similar = find_similar_questions(metadata["json_file_path"], metadata["cache_file"], i, top_k=-1)
        similar_examples = [x["index"] for x in similar[:num_examples]][::-1]
        dissimilar_examples = [x["index"] for x in similar[-num_examples:]]
        if i in similar_examples or i in dissimilar_examples:
            raise Exception("similar_examples or dissimilar_examples should not contain the current problem")
        similar_examples.append(i)
        dissimilar_examples.append(i)

        prompt_no_labels = utils.format_segments_math(dataset_full, [i])
        prompt_gt_labels = utils.format_segments_math(dataset_full, icl_examples)
        prompt_in_labels = utils.format_segments_math(dataset_full, icl_examples, replace_indices=replace_indices)
        prompt_si_labels = utils.format_segments_math(dataset_full, similar_examples)
        prompt_di_labels = utils.format_segments_math(dataset_full, dissimilar_examples)
        exp_names = ["no_labels", "correct_labels", "true_random_labels", "similar_labels", "dissimilar_labels"]
        prompts = [prompt_no_labels, prompt_gt_labels, prompt_in_labels, prompt_si_labels, prompt_di_labels]

        for exp_name, prompt in zip(exp_names, prompts):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are a math assistant."},
                          {"role": "user", "content": prompt}],
                max_tokens=4096,
            )
            data = {
                "idx": i,
                "answer": dataset_full[i]["answer_short"],
                "prompt": prompt,
                "experiment_name": exp_name,
                "response": response.choices[0].message.content.strip()
            }
            history.append(data)
        
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=4)

def run_experiment_math_local(dataset, dataset_full, model, tokenizer, device, num_examples, output_file, metadata):
    history = []

    example_rng = np.random.RandomState(1)
    num_problems = len(dataset_full)
    limit = min(len(dataset), 1000)

    for shuff_idx in tqdm(range(limit)):
        i = dataset[shuff_idx]["idx"]
        icl_examples = example_rng.choice(num_problems-1, num_examples, replace=False)
        icl_examples = [x if x < i else x+1 for x in icl_examples]
        if i in icl_examples:
            raise Exception("icl_examples should not contain the current problem")
        icl_examples.append(i)
        replace_indices = []
        for j in range(num_examples):
            idx = example_rng.randint(0, num_problems-1)
            idx = idx if idx < i else idx + 1
            replace_indices.append(idx)

        similar = find_similar_questions(metadata["json_file_path"], metadata["cache_file"], i, top_k=-1)
        similar_examples = [x["index"] for x in similar[:num_examples]][::-1]
        dissimilar_examples = [x["index"] for x in similar[-num_examples:]]
        if i in similar_examples or i in dissimilar_examples:
            raise Exception("similar_examples or dissimilar_examples should not contain the current problem")
        similar_examples.append(i)
        dissimilar_examples.append(i)

        prompt_no_labels = utils.format_segments_math(dataset_full, [i])
        prompt_gt_labels = utils.format_segments_math(dataset_full, icl_examples)
        prompt_in_labels = utils.format_segments_math(dataset_full, icl_examples, replace_indices=replace_indices)
        prompt_si_labels = utils.format_segments_math(dataset_full, similar_examples)
        prompt_di_labels = utils.format_segments_math(dataset_full, dissimilar_examples)
        exp_names = ["no_labels", "correct_labels", "true_random_labels", "similar_labels", "dissimilar_labels"]
        prompts = [prompt_no_labels, prompt_gt_labels, prompt_in_labels, prompt_si_labels, prompt_di_labels]

        for exp_name, prompt in zip(exp_names, prompts):
            response = utils.local_llm_query(model, tokenizer, device, prompt, max_new_tokens=2048)
            data = {
                "idx": i,
                "answer": dataset_full[i]["answer_short"],
                "prompt": prompt,
                "experiment_name": exp_name,
                "response": response
            }
            history.append(data)
        
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=4)

if __name__ == "__main__":
    json_file_path = "data/MATH/MATH.json"
    cache_file = "data/MATH/MATH_embeddings.npy"
    metadata = {
        "json_file_path": json_file_path,
        "cache_file": cache_file,
    }

    with open(json_file_path, 'r') as f:
        dataset = json.load(f)
        dataset_filtered = [x for x in dataset]
    
    dataset_filtered = [q for q in dataset]
    print(f"Difficulty-filtered dataset size: {len(dataset_filtered)}")
    shuffle = np.random.RandomState(42)
    shuffle.shuffle(dataset_filtered)
    dataset_filtered = dataset_filtered[:500]

    # num_examples = 4
    # output_file = f"output/MATH/MATH_qwen2.5-0.5b-instruct_k{num_examples}.json"
    # model_str = "Qwen/Qwen2.5-0.5B-Instruct"
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_str,
    #     torch_dtype="auto",
    #     device_map="auto"
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_str)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    # run_experiment_math_local(
    #     dataset=dataset_filtered,
    #     dataset_full=dataset,
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=device,
    #     num_examples=num_examples,
    #     output_file=output_file,
    #     metadata=metadata
    # )

    load_dotenv()
    client = OpenAI(api_key=os.getenv("API_KEY"))

    num_examples = 4
    model = "gpt-3.5-turbo"
    output_file = f"output/MATH/MATH_{model}_k{num_examples}.json"
    run_experiment_math_openai(
        client=client,
        dataset=dataset_filtered,
        dataset_full=dataset,
        model=model,
        num_examples=num_examples,
        output_file=output_file,
        metadata=metadata
    )