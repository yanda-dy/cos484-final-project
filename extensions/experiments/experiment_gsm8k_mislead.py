import json

from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv
import os

import utils
from similarity import find_similar_questions

def run_experiment_gsm8k_openai(client: OpenAI, dataset, dataset_wrong, model, num_examples, output_file, metadata):
    history = []

    example_rng = np.random.RandomState(1)
    num_problems = len(dataset)
    limit = min(num_problems, 1000)

    for i in tqdm(range(limit)):
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

        prompt_no_labels = utils.format_segments_gsm(dataset, [i])
        prompt_gt_labels = utils.format_segments_gsm(dataset, icl_examples)
        prompt_rd_labels = utils.format_segments_gsm(dataset, icl_examples, replace_indices=replace_indices)
        prompt_in_labels = utils.format_segments_gsm(dataset_wrong, icl_examples)
        prompt_qo_labels = utils.format_segments_gsm_custom(dataset, icl_examples, question_only=True)
        prompt_ao_labels = utils.format_segments_gsm_custom(dataset, icl_examples, answer_only=True)
        prompt_si_labels = utils.format_segments_gsm(dataset, similar_examples)
        prompt_di_labels = utils.format_segments_gsm(dataset, dissimilar_examples)
        exp_names = ["no_labels", "correct_labels", "true_random_labels", "incorrect_random_labels", "question_only", "answer_number_only", "similar_labels", "dissimilar_labels"]
        prompts = [prompt_no_labels, prompt_gt_labels, prompt_rd_labels, prompt_in_labels, prompt_qo_labels, prompt_ao_labels, prompt_si_labels, prompt_di_labels]
        
        for exp_name, prompt in zip(exp_names, prompts):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048
            )
            data = {
                "idx": i,
                "answer": dataset[i]["answer"].split()[-1],
                "prompt": prompt,
                "experiment_name": exp_name,
                "response": response.choices[0].message.content.strip()
            }
            history.append(data)
        
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=4)

def run_experiment_gsm8k_local(dataset, dataset_wrong, model, tokenizer, device, num_examples, output_file, metadata):
    history = []

    example_rng = np.random.RandomState(1)
    num_problems = len(dataset)
    limit = min(num_problems, 1000)

    for i in tqdm(range(limit)):
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

        prompt_rd_labels = utils.format_segments_gsm(dataset, icl_examples, replace_indices=replace_indices)
        prompt_in_labels = utils.format_segments_gsm(dataset_wrong, icl_examples)
        prompt_qo_labels = utils.format_segments_gsm_custom(dataset, icl_examples, question_only=True)
        prompt_ao_labels = utils.format_segments_gsm_custom(dataset, icl_examples, answer_only=True)
        exp_names = ["true_random_labels", "incorrect_random_labels", "question_only", "answer_number_only"]
        prompts = [prompt_rd_labels, prompt_in_labels, prompt_qo_labels, prompt_ao_labels]
        for exp_name, prompt in zip(exp_names, prompts):
            response = utils.local_llm_query(model, tokenizer, device, prompt, max_new_tokens=2048)
            data = {
                "idx": i,
                "answer": dataset[i]["answer"].split()[-1],
                "prompt": prompt,
                "experiment_name": exp_name,
                "response": response
            }
            history.append(data)
        
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=4)
        

if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(api_key=os.getenv("API_KEY"))

    json_file_path = "data/GSM8K/gsm8k_test.json"
    wrong_file_path = "data/GSM8K/gsm8k_test_incorrect.json"
    cache_file = "data/GSM8K/gsm8k_test_embeddings.npy"
    metadata = {
        "json_file_path": json_file_path,
        "cache_file": cache_file,
    }

    with open(json_file_path, 'r') as f:
        dataset = json.load(f)
    
    with open(wrong_file_path, 'r') as f:
        dataset_wrong = json.load(f)

    # model_str = "Qwen/Qwen2.5-0.5B-Instruct"
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_str,
    #     torch_dtype="auto",
    #     device_map="auto"
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_str)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")

    # num_examples = 4
    # output_file = f"output/GSM8K/gsm8k_test_qwen2.5-0.5b-instruct_k{num_examples}_mis.json"
    # run_experiment_gsm8k_local(
    #     dataset=dataset,
    #     dataset_wrong=dataset_wrong,
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=device,
    #     num_examples=num_examples,
    #     output_file=output_file,
    #     metadata=metadata
    # )

    num_examples = 4
    model = "gpt-4.1-mini"
    output_file = f"output/GSM8K/gsm8k_test_{model}_k{num_examples}_mis.json"
    run_experiment_gsm8k_openai(
        client=client,
        dataset=dataset,
        dataset_wrong=dataset_wrong,
        model=model,
        num_examples=num_examples,
        output_file=output_file,
        metadata=metadata
    )
