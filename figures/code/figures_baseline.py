import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scienceplots
plt.style.use(['science', 'ieee'])

datasets = ['Classification', 'Multiple Choice']
methods = ['no_labels', 'correct_labels', 'incorrect_random_labels']

method_names = {
    'no_labels': 'No Demos',
    'correct_labels': 'Demos w/ gold labels',
    'incorrect_random_labels': 'Demos w/ random labels'
}

short_names = {
    'no_labels': 'None',
    'correct_labels': 'Correct',
    'incorrect_random_labels': 'True Random'
}

# Main data
data = {
    'Classification': {
        'no_labels': 79.13,
        'correct_labels': 80.73,
        'incorrect_random_labels': 79.27
    },
    'Multiple Choice': {
        'no_labels': 85.13,
        'correct_labels': 87.20,
        'incorrect_random_labels': 87.47
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

fig, axes = plt.subplots(1, 2, figsize=(7.5, 4), sharey=False)

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

for i, dataset in enumerate(datasets):
    ax = axes[i]
    values = [data[dataset][method] for method in methods]
    
    # Plot bars
    bars = ax.bar(np.arange(len(methods)) * 1.1, values, width=1.0, color=[colors[m] for m in methods], linewidth=0)
    
    best_idx = methods.index(best_methods[dataset])
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)
    
    # Add value labels
    for j, bar in enumerate(bars):
        height = bar.get_height()
        weight = 'normal'
        
        # Improvement over baseline
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
                   ha='center', va='bottom', fontsize=14, weight=weight)
    
    ax.set_title(dataset, fontsize=22, fontweight='bold')
    ax.set_xticks([])
    
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
    ax.tick_params(axis='y', labelsize=14)
    
    buffer = 1
    y_min = min(values) - buffer
    y_max = max(values) + buffer
    if dataset == "GPT-3.5-turbo":
        y_min = 65.1
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    if i == 0:
        ax.set_ylabel('Accuracy (\%)', fontsize=17)

handles = [plt.Rectangle((0,0),1,1, color=colors[m]) for m in methods]
fig.legend(handles, [method_names[m] for m in methods], 
          loc='lower center', bbox_to_anchor=(0.55, -0.21), 
          ncol=2, fontsize=17)

max_improvement_dataset = max(improvements.items(), 
                             key=lambda x: max(x[1].values()))
max_improvement_method = max(max_improvement_dataset[1].items(),
                           key=lambda x: x[1])
max_improvement_value = max_improvement_method[1]

plt.tight_layout()
plt.savefig('figures/baseline.png', dpi=300, bbox_inches='tight')
