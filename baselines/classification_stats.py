import json
import csv
from collections import Counter, defaultdict

def compare_predictions(input_entry, pred_entry, model_keys):
    results = {}
    true_label = input_entry['true_label'].strip()

    for key in model_keys:
        pred_label = pred_entry.get(key, "").strip()
        correct = (pred_label == true_label)

        mismatch_type = None
        if not pred_label:
            mismatch_type = "empty"
        elif not correct:
            mismatch_type = "wrong"

        results[key] = {
            "prediction": pred_label,
            "correct": correct,
            "mismatch_type": mismatch_type
        }

    return results, true_label

def analyze_predictions(input_file, prediction_file):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        input_data = {json.loads(line)['id']: json.loads(line) for line in f_in}

    with open(prediction_file, 'r', encoding='utf-8') as f_out:
        pred_data = [json.loads(line) for line in f_out]

    model_keys = ['zero_shot_output', 'icl_random_output', 'icl_gold_output']
    stats = {key: Counter() for key in model_keys}

    for pred_entry in pred_data:
        ex_id = str(pred_entry['id'])
        input_entry = input_data.get(ex_id)
        if not input_entry:
            print(f"Warning: ID not in input file: {ex_id}")
            continue

        comparisons, true_label = compare_predictions(input_entry, pred_entry, model_keys)

        for key in model_keys:
            stats[key]['total'] += 1
            if comparisons[key]['correct']:
                stats[key]['correct'] += 1
            else:
                mismatch_type = comparisons[key]['mismatch_type']
                stats[key][mismatch_type] += 1

    return stats

def compute_average_accuracy():
    input_file = "combined_classification_stats.csv"
    output_file = "average_accuracy_summary.csv"

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = []

        for row in reader:
            model = row['experiment']
            acc_values = []
            for key, value in row.items():
                if 'accuracy (%)' in key:
                    try:
                        acc_values.append(float(value))
                    except ValueError:
                        pass
            avg_acc = sum(acc_values) / len(acc_values) if acc_values else 0.0
            results.append({'model': model, 'k=16 avg accuracy (%)': f"{avg_acc:.2f}"})

    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=['model', 'k=16 avg accuracy (%)'])
        writer.writeheader()
        for entry in results:
            writer.writerow(entry)

    print(f"Average accuracy summary output: {output_file}")


def combine_and_export_all():
    datasets = {
        "mrpc": ("processed_mrpc.jsonl", "baseline_mrpc_gpt-4.1-mini_k=16.jsonl"),
        "rte": ("processed_rte.jsonl", "baseline_rte_gpt-4.1-mini_k=16.jsonl"),
        "tweeteval_hate": ("processed_tweeteval_hate.jsonl", "baseline_tweeteval_hate_gpt-4.1-mini_k=16.jsonl")
    }

    all_stats = defaultdict(dict)
    model_keys = ['zero_shot_output', 'icl_random_output', 'icl_gold_output']

    for dataset_name, (input_path, pred_path) in datasets.items():
        stats = analyze_predictions(input_path, pred_path)
        for model in model_keys:
            counter = stats[model]
            total = counter['total']
            correct = counter['correct']
            acc = (correct / total * 100) if total else 0.0
            all_stats[model][dataset_name] = {
                'total': total,
                'correct': correct,
                'accuracy (%)': f"{acc:.2f}",
                'wrong': counter['wrong'],
                'empty': counter['empty']
            }

    # write csv
    output_file = "combined_classification_stats.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['experiment']
        for dataset in datasets.keys():
            fieldnames.extend([
                f"{dataset}_total",
                f"{dataset}_correct",
                f"{dataset}_accuracy (%)",
                f"{dataset}_wrong",
                f"{dataset}_empty"
            ])
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model, dataset_stats in all_stats.items():
            row = {'experiment': model}
            for dataset in datasets.keys():
                ds = dataset_stats.get(dataset, {})
                row.update({
                    f"{dataset}_total": ds.get('total', 0),
                    f"{dataset}_correct": ds.get('correct', 0),
                    f"{dataset}_accuracy (%)": ds.get('accuracy (%)', '0.00'),
                    f"{dataset}_wrong": ds.get('wrong', 0),
                    f"{dataset}_empty": ds.get('empty', 0),
                })
            writer.writerow(row)

    print(f"Combined stats output: {output_file}")


combine_and_export_all()
compute_average_accuracy()