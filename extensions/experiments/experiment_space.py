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

def run_experiment_csqa_local(dataset, wordbank, oodbank, model, tokenizer, device, num_examples, output_file, metadata, limit=1000):
    history = []

    example_rng = np.random.RandomState(1)
    num_problems = len(dataset)
    to_process = min(num_problems, limit)
    for i in tqdm(range(to_process)):
        icl_examples = example_rng.choice(num_problems-1, num_examples, replace=False)
        icl_examples = [x if x < i else x+1 for x in icl_examples]
        if i in icl_examples:
            raise Exception("icl_examples should not contain the current problem")
        icl_examples.append(i)

        prompt_baseline = utils.format_segments(dataset, [i])
        prompt_rd_english = utils.format_random(dataset, i, num_examples, wordbank, format=True)
        prompt_ood_english = utils.format_random(dataset, i, num_examples, wordbank, format=False)
        prompt_q_only = utils.format_only(dataset, icl_examples, question_only=True)
        prompt_a_only = utils.format_only(dataset, icl_examples, answer_only=True)
        prompt_ood_math = utils.format_ood_math(dataset, i, num_examples, oodbank)
        exp_names = ["baseline", "random_english", "ood_english", "question_only", "answer_only", "ood_math"]
        prompts = [prompt_baseline, prompt_rd_english, prompt_ood_english, prompt_q_only, prompt_a_only, prompt_ood_math]

        for exp_name, prompt in zip(exp_names, prompts):
            prompt = prompt.replace("A: ", "1: ").replace("B: ", "2: ").replace("C: ", "3: ").replace("D: ", "4: ").replace("E: ", "5: ")
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

def run_experiment_csqa_full(dataset, wordbank, oodbank, model, tokenizer, device, num_examples, output_file, metadata, limit=1000):
    history = []

    example_rng = np.random.RandomState(1)
    num_problems = len(dataset)
    to_process = min(num_problems, limit)
    for i in tqdm(range(to_process)):
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

        prompt_baseline = utils.format_segments(dataset, [i])
        prompt_gt_labels = utils.format_segments(dataset, icl_examples, show_labels=metadata["show_labels"])
        prompt_in_labels = utils.format_segments(dataset, icl_examples, correct=False, show_labels=metadata["show_labels"])
        prompt_si_labels = utils.format_segments(dataset, similar_examples, show_labels=metadata["show_labels"])
        prompt_di_labels = utils.format_segments(dataset, dissimilar_examples, show_labels=metadata["show_labels"])
        prompt_rd_english = utils.format_random(dataset, i, num_examples, wordbank, format=True)
        prompt_ood_english = utils.format_random(dataset, i, num_examples, wordbank, format=False)
        prompt_q_only = utils.format_only(dataset, icl_examples, question_only=True)
        prompt_a_only = utils.format_only(dataset, icl_examples, answer_only=True)
        prompt_ood_math = utils.format_ood_math(dataset, i, num_examples, oodbank)
        exp_names = ["baseline", "correct_examples", "incorrect_examples", "similar_examples", "dissimilar_examples",
                     "random_english", "ood_english", "question_only", "answer_only", "ood_math"]
        prompts = [prompt_baseline, prompt_gt_labels, prompt_in_labels, prompt_si_labels, prompt_di_labels,
                   prompt_rd_english, prompt_ood_english, prompt_q_only, prompt_a_only, prompt_ood_math]

        if i <= 666:
            continue

        for exp_name, prompt in zip(exp_names, prompts):
            prompt = prompt.replace("A: ", "1: ").replace("B: ", "2: ").replace("C: ", "3: ").replace("D: ", "4: ").replace("E: ", "5: ")
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
    json_file_path = "data/CSQA/csqa_train.json"
    cache_file = "data/CSQA/csqa_train_embeddings.npy"
    metadata = {
        "json_file_path": json_file_path,
        "cache_file": cache_file,
        "show_labels": True,
    }

    with open(json_file_path, 'r') as f:
        dataset = json.load(f)
    
    with open("data/scrabble_dictionary.txt", 'r') as f:
        wordbank = f.read().splitlines()
    
    with open("data/GSM8K/gsm8k_test.json", 'r') as f:
        oodbank = json.load(f)

    wordbank = [word.lower() for word in wordbank]

    # Load environment variables from a .env file
    load_dotenv()
    
    model_str = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    for num_examples in [16]:
        output_file = f"output/CSQA/csqa_train_llama-3.2-1b-instruct_k{num_examples}_full.json"
        print(f"Running experiment with {num_examples} example(s)")
        run_experiment_csqa_full(
            dataset=dataset,
            wordbank=wordbank,
            oodbank=oodbank,
            model=model,
            tokenizer=tokenizer,
            device=device,
            num_examples=num_examples,
            output_file=output_file,
            metadata=metadata,
            limit=1000
        )

