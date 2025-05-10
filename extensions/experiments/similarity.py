import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

def find_similar_questions(json_file_path, cache_file, question_index, top_k=5):
    with open(json_file_path, 'r') as f:
        questions = json.load(f)
    
    target_question = questions[question_index]
    target_type = target_question["type"]

    questions_filtered = [q for q in questions if q["type"] == target_type]
    question_texts = [q["question"] for q in questions_filtered]

    # print(f"Number of questions of type {target_type}: {len(question_texts)}")
    if top_k == -1:
        top_k = len(question_texts) - 1

    if os.path.exists(cache_file):
        embeddings = np.load(cache_file)
    else:
        raise Exception(f"Cache file not found: {cache_file}. Please run the embedding generation step first.")
    
    target_embedding = embeddings[question_index].reshape(1, -1)
    
    # cosine similarity
    similarities = cosine_similarity(target_embedding, embeddings).flatten()
    
    # exclude self
    similarities[question_index] = -1
    most_similar_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in most_similar_indices:
        results.append({
            "index": int(idx),
            "similarity_score": float(similarities[idx]),
            "question": questions[idx]["question"],
            "type": questions[idx]["type"]
        })
    
    return results

def generate_embeddings(json_file_path, model_name, cache_file):
    with open(json_file_path, 'r') as f:
        questions = json.load(f)
    
    question_texts = [q["question"] for q in questions]
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    embeddings = model.encode(question_texts, show_progress_bar=True, convert_to_numpy=True)
    np.save(cache_file, embeddings)

def main():
    json_file_path = "data/MATH/MATH.json"
    cache_file = "data/MATH/MATH_embeddings.npy"
    model_name = "math-similarity/Bert-MLM_arXiv-MP-class_zbMath"

    json_file_path = "data/ARC/arc_challenge_test.json"
    cache_file = "data/ARC/arc_challenge_test_embeddings.npy"
    model_name = "mixedbread-ai/mxbai-embed-large-v1"

    # json_file_path = "data/ARC/arc_easy_test.json"
    # cache_file = "data/ARC/arc_easy_test_embeddings.npy"
    # model_name = "mixedbread-ai/mxbai-embed-large-v1"

    # json_file_path = "data/GSM8K/gsm8k_test.json"
    # cache_file = "data/GSM8K/gsm8k_test_embeddings.npy"
    # model_name = "math-similarity/Bert-MLM_arXiv-MP-class_zbMath"
    
    # json_file_path = "data/CSQA/csqa_train.json"
    # cache_file = "data/CSQA/csqa_train_embeddings.npy"
    # model_name = "mixedbread-ai/mxbai-embed-large-v1"

    question_index = 0
    top_k = 5

    if not os.path.exists(cache_file):
        print("Embeddings cache not found. Generating embeddings...")
        generate_embeddings(json_file_path, model_name=model_name, cache_file=cache_file)
        print(f"Embeddings saved to {cache_file}")
    
    similar_questions = find_similar_questions(json_file_path, cache_file, question_index, -1)
    
    dataset = json.load(open(json_file_path, 'r'))
    target_question = dataset[question_index]
    print(f"Target question {question_index}:")
    print(f"Type: {target_question['type']}")
    print(f"Question: {target_question['question']}")
    print("Choices:")
    for label, choice in zip(target_question["choices"]["label"], target_question["choices"]["text"]):
        print(f"{label}: {choice}")
    print("="*80)
    
    for i, result in enumerate(similar_questions[:top_k]):
        print(f"Rank {i+1}: Question {result['index']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"Type: {result['type']}")
        print(f"Question: {result['question']}")
        print("Choices:")
        ref_q = dataset[result["index"]]
        for label, choice in zip(ref_q["choices"]["label"], ref_q["choices"]["text"]):
            print(f"{label}: {choice}")
        print("-"*80)
    
    print("="*80)
    print("="*80)

    for i, result in enumerate(similar_questions[-top_k:][::-1]):
        print(f"Rank -{i+1}: Question {result['index']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"Type: {result['type']}")
        print(f"Question: {result['question']}")
        print("Choices:")
        ref_q = dataset[result["index"]]
        for label, choice in zip(ref_q["choices"]["label"], ref_q["choices"]["text"]):
            print(f"{label}: {choice}")
        print("-"*80)

if __name__ == "__main__":
    main()
