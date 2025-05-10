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

def run_experiment_arc_openai(client, dataset, model, num_examples, output_file, metadata):
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

        similar = find_similar_questions(metadata["json_file_path"], metadata["cache_file"], i, top_k=-1)
        similar_examples = [x["index"] for x in similar[:num_examples]][::-1]
        dissimilar_examples = [x["index"] for x in similar[-num_examples:]]
        if i in similar_examples or i in dissimilar_examples:
            raise Exception("similar_examples or dissimilar_examples should not contain the current problem")
        similar_examples.append(i)
        dissimilar_examples.append(i)

        prompt_no_labels = utils.format_segments(dataset, [i])
        prompt_gt_labels = utils.format_segments(dataset, icl_examples)
        prompt_in_labels = utils.format_segments(dataset, icl_examples, correct=False)
        prompt_si_labels = utils.format_segments(dataset, similar_examples)
        prompt_di_labels = utils.format_segments(dataset, dissimilar_examples)
        exp_names = ["no_labels", "random_labels", "incorrect_labels", "similar_labels", "dissimilar_labels"]
        prompts = [prompt_no_labels, prompt_gt_labels, prompt_in_labels, prompt_si_labels, prompt_di_labels]

        for exp_name, prompt in zip(exp_names, prompts):
            response = utils.llm_query(client, model, prompt)
            data = {
                "idx": i,
                "id": dataset[i]["id"],
                "answerKey": dataset[i]["answerKey"],
                "prompt": prompt,
                "experiment_name": exp_name,
                "response": response,
            }
            history.append(data)
        
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=4)

def run_experiment_arc_local(dataset, model, tokenizer, device, num_examples, output_file, metadata):
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

        similar = find_similar_questions(metadata["json_file_path"], metadata["cache_file"], i, top_k=-1)
        similar_examples = [x["index"] for x in similar[:num_examples]][::-1]
        dissimilar_examples = [x["index"] for x in similar[-num_examples:]]
        if i in similar_examples or i in dissimilar_examples:
            raise Exception("similar_examples or dissimilar_examples should not contain the current problem")
        similar_examples.append(i)
        dissimilar_examples.append(i)

        prompt_no_labels = utils.format_segments(dataset, [i])
        prompt_gt_labels = utils.format_segments(dataset, icl_examples)
        prompt_in_labels = utils.format_segments(dataset, icl_examples, correct=False)
        prompt_si_labels = utils.format_segments(dataset, similar_examples)
        prompt_di_labels = utils.format_segments(dataset, dissimilar_examples)
        exp_names = ["no_labels", "random_labels", "incorrect_labels", "similar_labels", "dissimilar_labels"]
        prompts = [prompt_no_labels, prompt_gt_labels, prompt_in_labels, prompt_si_labels, prompt_di_labels]

        for exp_name, prompt in zip(exp_names, prompts):
            response = utils.local_llm_query(model, tokenizer, device, prompt)
            data = {
                "idx": i,
                "id": dataset[i]["id"],
                "answerKey": dataset[i]["answerKey"],
                "prompt": prompt,
                "experiment_name": exp_name,
                "response": response,
            }
            history.append(data)
        
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=4)

if __name__ == "__main__":
    json_file_path = "data/ARC/arc_challenge_test.json"
    cache_file = "data/ARC/arc_challenge_test_embeddings.npy"
    metadata = {
        "json_file_path": json_file_path,
        "cache_file": cache_file,
        "show_labels": True,
    }

    with open(json_file_path, 'r') as f:
        dataset = json.load(f)

    # Load environment variables from a .env file
    load_dotenv()

    # client = OpenAI(api_key=os.getenv("API_KEY"))
    # model = "gpt-3.5-turbo"
    # run_experiment_arc_openai(
    #     client=client,
    #     dataset=dataset,
    #     model=model,
    #     num_examples=4,
    #     output_file="output/ARC/arc_challenge_test.json",
    #     metadata=metadata
    # )

    num_examples = 16
    output_file = f"output/ARC/arc_challenge_test_qwen2.5-0.5b-instruct_k{num_examples}.json"

    model_str = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_experiment_arc_local(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_examples=num_examples,
        output_file=output_file,
        metadata=metadata
    )
