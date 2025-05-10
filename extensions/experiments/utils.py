import random
from openai import OpenAI

def llm_query(client: OpenAI, model, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def local_llm_query(model, tokenizer, device, prompt, max_new_tokens=2048):
    try:
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        attention_mask = model_inputs["attention_mask"]

        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_k=None,
            top_p=None,
            temperature=None,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        return f"ERROR: {str(e)}"

def format_example_arc(problem, index, correct=True, show_labels=True):
    question = problem["question"]
    choices = problem["choices"]
    answer_key = problem["answerKey"]

    corresp = {}
    example = [f"Example {index}", "Question:", question, "Choices:"]
    for choice, label in zip(choices['text'], choices['label']):
        corresp[label] = choice
        if show_labels:
            example.append(f"{label}: {choice}")
        else:
            example.append(f"{choice}")

    example.append("Answer:")
    if correct:
        if show_labels:
            example.append(f"{answer_key}: {corresp[answer_key]}")
        else:
            example.append(f"{corresp[answer_key]}")
    else:
        answers = [label for label in choices['label'] if label != answer_key]
        option = random.choice(answers)
        if show_labels:
            example.append(f"{option}: {corresp[option]}")
        else:
            example.append(f"{corresp[option]}")
    formatted_example = "\n".join(example)
    return formatted_example

def format_prompt_arc(problem, show_labels=True):
    question = problem["question"]
    choices = problem["choices"]
    answer_key = problem["answerKey"]

    # prompt = ["Now, answer the following multiple-choice question. Output only the line corresponding to your selection. Your response should be in the format '[letter choice]: [choice text]'.", "Question:", question, "Choices:"]
    # prompt = ["Now, answer the following multiple-choice question. Output only the line corresponding to your selection.", "Question:", question, "Choices:"]
    prompt = ["Now, answer the following multiple-choice question. Output only the line corresponding to your selection, which MUST BE IN THE FORMAT `Answer: number: text`.", "Question:", question, "Choices:"]
    for choice, label in zip(choices['text'], choices['label']):
        if choice == answer_key:
            answer_key = label
        if show_labels:
            prompt.append(f"{label}: {choice}")
        else:
            prompt.append(f"{choice}")

    prompt.append("Answer:\n")
    formatted_prompt = "\n".join(prompt)
    return formatted_prompt

def format_segments(dataset, problems, correct=True, show_labels=True):
    if len(problems) == 1:
        return format_prompt_arc(dataset[problems[0]])
    
    else:
        in_context = problems[:-1]
        segments = []
        for i, problem in enumerate(in_context):
            segments.append(format_example_arc(dataset[problem], i + 1, correct=correct, show_labels=show_labels))
        segments.append(format_prompt_arc(dataset[problems[-1]], show_labels=show_labels))
        return "\n\n".join(segments)

def evaluate_response(response, ref_answer):
    response = response.split("\n")[-1].strip().upper()
    if response == ref_answer:
        return True
    else:
        return False

def format_prompt_gsm(problem, prompt_inject="Now, reason through and answer the following math question."):
    question = problem["question"]
    prompt = [prompt_inject, "Question:", question, "Answer:"]
    prompt.append("")

    formatted_prompt = "\n".join(prompt)
    return formatted_prompt

def format_example_gsm(problem, index):
    question = problem["question"]
    answer = problem["answer"]
    example = [f"Example {index}", "Question:", question, "Answer:", answer]
    formatted_example = "\n".join(example)
    return formatted_example

def format_incorrect_example_gsm(problem, replace, index):
    question = problem["question"]
    answer = replace["answer"]
    example = [f"Example {index}", "Question:", question, "Answer:", answer]
    formatted_example = "\n".join(example)
    return formatted_example

def format_segments_gsm(dataset, problems, replace_indices=None):
    if len(problems) == 1:
        return format_prompt_gsm(dataset[problems[0]])
    
    else:
        in_context = problems[:-1]
        segments = []
        for i, problem in enumerate(in_context):
            if replace_indices is None:
                segments.append(format_example_gsm(dataset[problem], i + 1))
            else:
                segments.append(format_incorrect_example_gsm(dataset[problem], dataset[replace_indices[i]], i + 1))
        segments.append(format_prompt_gsm(dataset[problems[-1]]))
        return "\n\n".join(segments)

def format_segments_gsm_custom(dataset, problems, question_only=False, answer_only=False):
    if len(problems) == 1:
        return format_prompt_gsm(dataset[problems[0]])
    
    else:
        in_context = problems[:-1]
        segments = []
        for i, pidx in enumerate(in_context):
            problem = dataset[pidx]
            question, answer = problem["question"], problem["answer"]
            if question_only:
                fmt = '\n'.join([f"Example {i+1}", "Question:", question])
                segments.append(fmt)
            elif answer_only:
                answer = problem["answer"].split()[-1]
                fmt = '\n'.join([f"Example {i+1}", "Question:", question, "Answer:", answer])
                segments.append(fmt)
            else:
                raise Exception("question_only and answer_only cannot be both False")
        segments.append(format_prompt_gsm(dataset[problems[-1]]))
        return "\n\n".join(segments)

def format_segments_math(dataset, problems, replace_indices=None):
    prompt_inject = "Now, reason through and answer the following math question. Make sure to box your final answer in LaTeX with \\boxed{answer}"
    if len(problems) == 1:
        return format_prompt_gsm(dataset[problems[0]], prompt_inject=prompt_inject)
    
    else:
        in_context = problems[:-1]
        segments = []
        for i, problem in enumerate(in_context):
            if replace_indices is None:
                segments.append(format_example_gsm(dataset[problem], i + 1))
            else:
                segments.append(format_incorrect_example_gsm(dataset[problem], dataset[replace_indices[i]], i + 1))
        segments.append(format_prompt_gsm(dataset[problems[-1]], prompt_inject=prompt_inject))
        return "\n\n".join(segments)

def format_random(dataset, problem, num_examples, wordbank, format=True):
    segments = []
    for i in range(num_examples):
        num_words = random.randint(10, 20)
        phrase = []
        for _ in range(num_words):
            word = random.choice(wordbank)
            phrase.append(word)
        phrase = " ".join(phrase)
        temp = [f"Example {i+1}", "Question:", phrase]

        if format:
            temp.append("Choices:")
            for c in "ABCDE":
                num_words = random.randint(1, 3)
                phrase = []
                for _ in range(num_words):
                    word = random.choice(wordbank)
                    phrase.append(word)
                phrase = " ".join(phrase)
                temp.append(f"{c}: {phrase}")
        
        temp.append("Answer:")
        option = random.choice(list("ABCDE"))
        phrase = []
        for _ in range(num_words):
            word = random.choice(wordbank)
            phrase.append(word)
        phrase = " ".join(phrase)
        temp.append(f"{option}: {phrase}")

        example = '\n'.join(temp)
        segments.append(example)
    
    segments.append(format_prompt_arc(dataset[problem]))
    return "\n\n".join(segments)

def format_only(dataset, problems, question_only=False, answer_only=False):
    assert question_only != answer_only, "Only one of question_only or answer_only must be true."

    segments = []
    for i, pidx in enumerate(problems[:-1]):
        problem = dataset[pidx]
        question = problem["question"]
        choices = problem["choices"]
        answer_key = problem["answerKey"]
        if question_only:
            prompt = [f"Example {i+1}", "Question:", question]
            for choice, label in zip(choices['text'], choices['label']):
                prompt.append(f"{label}: {choice}")
            fmt = '\n'.join(prompt)
        elif answer_only:
            prompt = [f"Example {i+1}", "Answer:"]
            answer = ''
            for choice, label in zip(choices['text'], choices['label']):
                if label == answer_key:
                    answer = choice
            prompt.append(f"{answer_key}: {answer}")
            fmt = '\n'.join(prompt)
        segments.append(fmt)
    segments.append(format_prompt_arc(dataset[problems[-1]]))
    return "\n\n".join(segments)

def format_ood_math(dataset, problem, num_problems, oodbank):
    segments = []
    for i in range(num_problems):
        problem_idx = random.randint(0, len(oodbank) - 1)
        ood_problem = oodbank[problem_idx]
        question = ood_problem["question"]
        answer = ood_problem["answer"]
        example = [f"Example {i+1}", "Question:", question, "Answer:", answer]
        segments.append('\n'.join(example))
    
    segments.append(format_prompt_arc(dataset[problem]))
    return "\n\n".join(segments)