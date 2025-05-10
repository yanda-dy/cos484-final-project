import numpy as np
import json
import matplotlib.pyplot as plt

files = {
    'Balanced COPA': 'icl_question_vs_answer_balanced_copa_gpt-4.1-mini_k=16.jsonl',
    'Commonsense QA': 'icl_question_vs_answer_CommonsenseQA_gpt-4.1-mini_k=16.jsonl',
    'Openbook QA': 'icl_question_vs_answer_openbookqa_gpt-4.1-mini_k=16.jsonl'
}

model_keys = ['zero_shot_output', 'icl_questions_only_output', 'icl_answers_only_output']
labels_map = {
    'zero_shot_output': 'Zero-Shot',
    'icl_questions_only_output': 'Questions-Only',
    'icl_answers_only_output': 'Answers-Only'
}

def extract_label(text):
    return text.strip().split(":")[0] if ":" in text else text.strip()

accuracy_data = {k: [] for k in model_keys}
file_labels = []

for file_label, file_path in files.items():
    file_labels.append(file_label)
    correct_counts = {k: 0 for k in model_keys}
    total = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            true = entry['true_answer']['label'].strip()
            total += 1
            for key in model_keys:
                pred_raw = entry.get(key, "")
                pred_label = extract_label(pred_raw)
                if pred_label == true:
                    correct_counts[key] += 1

    for key in model_keys:
        accuracy = correct_counts[key] / total * 100 if total else 0.0
        accuracy_data[key].append(accuracy)

x = np.arange(len(file_labels))
bar_width = 0.25
gap = 0.03

fig, ax = plt.subplots()

for i, key in enumerate(model_keys):
    offset = i * (bar_width + gap)
    ax.bar(x + offset, accuracy_data[key], width=bar_width, label=labels_map[key])

ax.set_xlabel('Dataset')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy Comparison')
total_width = len(model_keys) * (bar_width + gap) - gap
ax.set_xticks(x + total_width / 2 - bar_width / 2)
ax.set_xticklabels(file_labels)
ax.set_ylim(bottom=75, top=95)
ax.legend(title='Model Type')

plt.tight_layout()
plt.show()
