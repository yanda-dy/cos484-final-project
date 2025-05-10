import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scienceplots
plt.style.use(['science', 'ieee'])

datasets = ['GPT-3.5-turbo', 'Qwen2.5-0.5b']
methods = ['no_labels', 'correct_labels', 'true_random_labels', 'incorrect_random_labels', 
           'question_only', 'answer_number_only', 'similar_labels', 'dissimilar_labels']

method_names = {
    'no_labels': 'No Examples',
    'correct_labels': 'Correct Examples',
    'true_random_labels': 'Random Labels',
    'incorrect_random_labels': 'Incorrect Examples',
    'question_only': 'Question Only',
    'answer_number_only': 'Final Answer Only',
    'similar_labels': 'Similar Examples',
    'dissimilar_labels': 'Dissimilar Examples'
}

short_names = {
    'no_labels': 'None',
    'correct_labels': 'Correct',
    'true_random_labels': 'True Random',
    'incorrect_random_labels': 'Incorrect',
    'question_only': 'Question Only',
    'answer_number_only': 'Final Answer Only',
    'similar_labels': 'Similar',
    'dissimilar_labels': 'Dissimilar'
}

# Main data
data = {
    'GPT-3.5-turbo': {
        'no_labels': 68.50,
        'correct_labels': 73.10,
        'true_random_labels': 71.30,
        'incorrect_random_labels': 76.00,
        'question_only': 67.30,
        'answer_number_only': 31.50,
        'similar_labels': 73.30,
        'dissimilar_labels': 73.80
    },
    'Qwen2.5-0.5b': {
        'no_labels': 36.80,
        'correct_labels': 41.00,
        'true_random_labels': 41.10,
        'incorrect_random_labels': 40.90,
        'question_only': 39.00,
        'answer_number_only': 42.00,
        'similar_labels': 41.10,
        'dissimilar_labels': 39.40
    }
}

best_methods = {}
for dataset in datasets:
    best_method = max(data[dataset], key=data[dataset].get)
    best_methods[dataset] = best_method

improvements = {}
for dataset in datasets:
    baseline = data[dataset]['no_labels']
    improvements[dataset] = {
        method: data[dataset][method] - baseline for method in methods if method != 'no_labels'
    }

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)

colors = {
    'no_labels': '#80b2cf',
    'correct_labels': '#ffce73',
    'true_random_labels': '#f9b282',
    'incorrect_random_labels': '#f58676',
    'question_only': '#94d0c0',
    'answer_number_only': '#a3a7e2',
    'similar_labels': '#4EAA73',
    'dissimilar_labels': '#E38DBC'
}

# Plot each dataset in its own subplot
for i, dataset in enumerate(datasets):
    ax = axes[i]
    values = [data[dataset][method] for method in methods]
    bars = ax.bar(np.arange(len(methods)) * 1.1, values, width=1.0, color=[colors[m] for m in methods], linewidth=0)
    
    best_idx = methods.index(best_methods[dataset])
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)
    
    # Add value labels
    for j, bar in enumerate(bars):
        height = bar.get_height()
        weight = 'normal'
        
        if methods[j] != 'no_labels':
            improvement = height - data[dataset]['no_labels']
            add = '+' if improvement > 0 else '-'
            label_text = f'{height:.1f}\n{add}{improvement:.1f}'
        else:
            label_text = f'{height:.1f}'
        
        if height < bars[0].get_height() - 5:
            height = bars[0].get_height() - 3.3
        ax.annotate(label_text,
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, weight=weight)
    
    ax.set_title(dataset, fontsize=16, fontweight='bold')
    ax.set_xticks([])
    
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
    ax.tick_params(axis='y', labelsize=10)

    buffer = 2
    y_min = min(values) - buffer
    y_max = max(values) + buffer
    if dataset == "GPT-3.5-turbo":
        y_min = 65.1
    ax.set_ylim(y_min, y_max)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    if i == 0:
        ax.set_ylabel('Accuracy (\%)', fontsize=14)

handles = [plt.Rectangle((0,0),1,1, color=colors[m]) for m in methods]
fig.legend(handles, [method_names[m] for m in methods], 
          loc='right', bbox_to_anchor=(1.27, 0.55), 
          ncol=1, fontsize=14)

plt.tight_layout()
plt.savefig('figures/reasoning_gsm8k_comparison.png', dpi=300, bbox_inches='tight')
