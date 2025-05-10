import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scienceplots
plt.style.use(['science', 'ieee'])

datasets = ['CSQA', 'ARC_easy', 'ARC_challenge']
methods = ['no_labels', 'random_labels', 'incorrect_labels', 'similar_labels', 'dissimilar_labels']

method_names = {
    'no_labels': 'No Examples',
    'random_labels': 'Correct Examples',
    'incorrect_labels': 'Incorrect Examples',
    'similar_labels': 'Similar Examples',
    'dissimilar_labels': 'Dissimilar Examples'
}

short_names = {
    'no_labels': 'None',
    'random_labels': 'Random',
    'incorrect_labels': 'Incorrect',
    'similar_labels': 'Similar',
    'dissimilar_labels': 'Dissimilar'
}

# Main data
data = {
    'CSQA': {
        'no_labels': 48.70,
        'random_labels': 54.10,
        'incorrect_labels': 53.50,
        'similar_labels': 54.60,
        'dissimilar_labels': 54.20
    },
    'ARC_easy': {
        'no_labels': 65.49,
        'random_labels': 69.48,
        'incorrect_labels': 68.85,
        'similar_labels': 71.57,
        'dissimilar_labels': 69.03
    },
    'ARC_challenge': {
        'no_labels': 46.20,
        'random_labels': 50.30,
        'incorrect_labels': 50.60,
        'similar_labels': 52.00,
        'dissimilar_labels': 51.80
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

fig, axes = plt.subplots(3, 1, figsize=(4.5, 6.7), sharey=False)

colors = {
    'no_labels': '#80b2cf',
    'random_labels': '#ffce73',
    'incorrect_labels': '#f58676',
    'similar_labels': '#4EAA73',
    'dissimilar_labels': '#E38DBC'
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
            label_text = f'{height:.1f}\n+{improvement:.1f}'
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
    y_min = min(values) - buffer
    y_max = max(values) + buffer
    ax.set_ylim(y_min, y_max)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.text(-0.03, 0.48, 'Accuracy (\%)', va='center', rotation='vertical', fontsize=14)

handles = [plt.Rectangle((0,0),1,1, color=colors[m]) for m in methods]
fig.legend(handles, [method_names[m] for m in methods], 
          loc='lower center', bbox_to_anchor=(0.5, -0.14), 
          ncol=2, fontsize=14)

plt.tight_layout()
plt.savefig('figures/qwen_arc_vert.png', dpi=300, bbox_inches='tight')
