import json
import csv
from collections import Counter, defaultdict

def parse_label_text(answer_str):
    if ':' not in answer_str:
        return answer_str.strip(), ''
    label, text = answer_str.split(':', 1)
    return label.strip(), text.strip()

def compare_predictions(input_entry, pred_entry, model_keys):
    results = {}
    true_label = input_entry['true_answer']['label'].strip()
    true_text = input_entry['true_answer']['text'].strip()
    true_combined = f"{true_label}: {true_text}"
    valid_choices = [f"{c['label']}: {c['text']}" for c in input_entry.get("choices", [])]

    for key in model_keys:
        pred_raw = pred_entry.get(key, "").strip()
        pred_label, pred_text = parse_label_text(pred_raw)
        pred_combined = f"{pred_label}: {pred_text}"

        correct = (pred_combined == true_combined)
        mismatch_type = None

        if not pred_label or not pred_text:
            mismatch_type = "malformed"
        elif pred_combined not in valid_choices:
            mismatch_type = "not_in_choices"
        elif not correct:
            mismatch_type = "wrong"

        results[key] = {
            "prediction": pred_combined,
            "correct": correct,
            "mismatch_type": mismatch_type
        }

    return results

def evaluate_mcq(input_file, prediction_file):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        input_data = {json.loads(line)['question']: json.loads(line) for line in f_in}

    with open(prediction_file, 'r', encoding='utf-8') as f_out:
        pred_data = [json.loads(line) for line in f_out]

    model_keys = ['zero_shot_output', 'icl_random_output', 'icl_gold_output']
    stats = {key: Counter() for key in model_keys}

    for pred_entry in pred_data:
        question = pred_entry['question']
        input_entry = input_data.get(question)
        if not input_entry:
            print(f"Warning: Question not found in input: {question}")
            continue

        comparisons = compare_predictions(input_entry, pred_entry, model_keys)
        for key in model_keys:
            stats[key]['total'] += 1
            if comparisons[key]['correct']:
                stats[key]['correct'] += 1
            else:
                mismatch_type = comparisons[key]['mismatch_type']
                stats[key][mismatch_type] += 1

    return stats

def write_combined_csv(result_map, output_file):
    model_keys = ['zero_shot_output', 'icl_random_output', 'icl_gold_output']
    
    all_col_ids = set()
    for dataset_key, model_dict in result_map.items():
        for model in model_keys:
            if model in model_dict:
                all_col_ids.add(dataset_key)

    fieldnames = ['model']
    metric_suffixes = ['total', 'correct', 'accuracy (%)']
    for col_id in sorted(all_col_ids):
        for suffix in metric_suffixes:
            fieldnames.append(f"{col_id}_{suffix}")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in model_keys:
            row = {'model': model}
            for col_id in sorted(all_col_ids):
                stats = result_map.get(col_id, {}).get(model, {})
                total = stats.get('total', 0)
                correct = stats.get('correct', 0)
                acc = (correct / total * 100) if total else 0.0
                row.update({
                    f"{col_id}_total": total,
                    f"{col_id}_correct": correct,
                    f"{col_id}_accuracy (%)": f"{acc:.2f}"
                })
            writer.writerow(row)

    print(f"CSV output: {output_file}")

def write_averaged_accuracies_by_model(result_map, output_file):
    ks = [1, 4, 16, 32]
    datasets = ['CommonsenseQA', 'balanced_copa', 'openbookqa']
    model_keys = ['icl_random_output', 'icl_gold_output']

    rows = []
    for model in model_keys:
        row = {'model': model}
        for k in ks:
            accuracies = []
            for ds in datasets:
                label = f"{ds}_k={k}"
                stats = result_map.get(label, {}).get(model, {})
                total = stats.get('total', 0)
                correct = stats.get('correct', 0)
                if total > 0:
                    acc = correct / total * 100
                    accuracies.append(acc)
            avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
            row[f"k={k} avg accuracy (%)"] = f"{avg_acc:.2f}"
        rows.append(row)

    fieldnames = ['model'] + [f"k={k} avg accuracy (%)" for k in ks]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Averaged accuracy CSV output: {output_file}")


def write_overall_accuracies_by_model(result_map, output_file):
    ks = [16]
    datasets = ['CommonsenseQA', 'balanced_copa', 'openbookqa']
    model_keys = ['zero_shot_output', 'icl_random_output', 'icl_gold_output']

    rows = []
    for model in model_keys:
        row = {'model': model}
        for k in ks:
            accuracies = []
            for ds in datasets:
                label = f"{ds}_k={k}"
                stats = result_map.get(label, {}).get(model, {})
                total = stats.get('total', 0)
                correct = stats.get('correct', 0)
                if total > 0:
                    acc = correct / total * 100
                    accuracies.append(acc)
            avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
            row[f"k={k} avg accuracy (%)"] = f"{avg_acc:.2f}"
        rows.append(row)

    fieldnames = ['model'] + [f"k={k} avg accuracy (%)" for k in ks]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Averaged accuracy CSV output: {output_file}")

def run_mcq_evaluations():
    datasets = ['CommonsenseQA', 'balanced_copa', 'openbookqa']
    ks = [1, 4, 16, 32]

    k_result_map = defaultdict(dict)
    for k in ks:
        for ds in datasets:
            pred_file = f"baseline_{ds}_gpt-4.1-mini_k={k}.jsonl"
            input_file = f"processed_{ds}.jsonl"
            label = f"{ds}_k={k}"
            stats = evaluate_mcq(input_file, pred_file)
            for model in stats:
                k_result_map[label][model] = stats[model]

    write_combined_csv(k_result_map, "mcq_baseline_k_comparison.csv")

    write_averaged_accuracies_by_model(k_result_map, "mcq_averaged_accuracies.csv")

    write_overall_accuracies_by_model(k_result_map, "mcq_overall_accuracies.csv")
    # System prompt vs ICL for k=16
    sys_result_map = defaultdict(dict)
    for ds in datasets:
        input_file = f"processed_{ds}.jsonl"

        icl_pred = f"baseline_{ds}_gpt-4.1-mini_k=16.jsonl"
        sys_pred = f"system_{ds}_gpt-4.1-mini_k=16.jsonl"

        sys_result_map[f"{ds}_icl"] = evaluate_mcq(input_file, icl_pred)
        sys_result_map[f"{ds}_system"] = evaluate_mcq(input_file, sys_pred)

    write_combined_csv(sys_result_map, "mcq_system_vs_icl_k=16.csv")

run_mcq_evaluations()
