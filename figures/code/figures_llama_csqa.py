import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scienceplots
plt.style.use(['science', 'ieee'])

datasets = ['Qwen2.5-0.5b', 'Llama-3.2-1b']
methods = ['no_labels', 'correct_examples', 'incorrect_examples', 'similar_examples', 'dissimilar_examples',
           'question_only', 'answer_only', 'random_english', 'ood_english', 'ood_math']

method_names = {
    'no_labels': 'No Examples',
    'correct_examples': 'Correct Examples',
    'incorrect_examples': 'Incorrect Examples',
    'similar_examples': 'Similar Examples',
    'dissimilar_examples': 'Dissimilar Examples',
    'question_only': 'Question Only',
    'answer_only': 'Answer Only',
    'random_english': 'Random English$^1$',
    'ood_english': 'Random English$^2$',
    'ood_math': 'Random Math$^3$'
}

short_names = {
    'no_labels': 'No Examples',
    'correct_examples': 'Correct Examples',
    'incorrect_examples': 'Incorrect Examples',
    'similar_examples': 'Similar Examples',
    'dissimilar_examples': 'Dissimilar Examples',
    'question_only': 'Question Only',
    'answer_only': 'Answer Only',
    'random_english': 'Random English$^1$',
    'ood_english': 'Random English$^2$',
    'ood_math': 'Random Math$^3$'
}

# Main data
data = {
    'Qwen2.5-0.5b': {
        'no_labels': 48.70,
        'correct_examples': 53.90,
        'incorrect_examples': 53.50,
        'similar_examples': 54.60,
        'dissimilar_examples': 54.20,
        'question_only': 52.30,
        'answer_only': 47.50,
        'random_english': 51.40,
        'ood_english': 43.80,
        'ood_math': 49.00
    },
    'Llama-3.2-1b': {
        'no_labels': 41.60,
        'correct_examples': 49.70,
        'incorrect_examples': 50.30,
        'similar_examples': 48.90,
        'dissimilar_examples': 50.00,
        'question_only': 26.10,
        'answer_only': 32.30,
        'random_english': 37.10,
        'ood_english': 32.10,
        'ood_math': 45.70
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
    'correct_examples': '#ffce73',
    'incorrect_examples': '#f58676',
    'question_only': '#94d0c0',
    'answer_only': '#a3a7e2',
    'similar_examples': '#4EAA73',
    'dissimilar_examples': '#E38DBC',
    'random_english': '#cdb4db',
    'ood_english': '#ffc8dd',
    'ood_math': '#a2d2ff'
}

for i, dataset in enumerate(datasets):
    ax = axes[i]
    values = [data[dataset][method] for method in methods]
    
    # bar graph
    bars = ax.bar(np.arange(len(methods)) * 1.1, values, width=1.0, color=[colors[m] for m in methods], linewidth=0)
    
    best_idx = methods.index(best_methods[dataset])
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)
    for j, bar in enumerate(bars):
        height = bar.get_height()
        weight = 'normal'
        
        if methods[j] != 'no_labels':
            improvement = height - data[dataset]['no_labels']
            add = '+' if improvement > 0 else '-'
            label_text = f'{height:.1f}\n{add}{improvement:.1f}'
        else:
            label_text = f'{height:.1f}'
        
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
    if dataset == 'Llama-3.2-1b':
        buffer = 4
    y_min = min(values) - buffer
    y_max = max(values) + buffer
    ax.set_ylim(y_min, y_max)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
fig.text(-0.03, 0.5, 'Accuracy (\%)', va='center', rotation='vertical', fontsize=14)

handles = [plt.Rectangle((0,0),1,1, color=colors[m]) for m in methods]
fig.legend(handles, [method_names[m] for m in methods], 
          loc='right', bbox_to_anchor=(1.27, 0.48), 
          ncol=1, fontsize=14)

plt.tight_layout()
plt.savefig('figures/llama_qwen_csqa.png', dpi=300, bbox_inches='tight')
