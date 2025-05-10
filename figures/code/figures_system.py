import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
from io import StringIO
import scienceplots
plt.style.use(['science', 'ieee'])

csv_data = """model,CommonsenseQA_icl_total,CommonsenseQA_icl_correct,CommonsenseQA_icl_accuracy (%),CommonsenseQA_system_total,CommonsenseQA_system_correct,CommonsenseQA_system_accuracy (%),balanced_copa_icl_total,balanced_copa_icl_correct,balanced_copa_icl_accuracy (%),balanced_copa_system_total,balanced_copa_system_correct,balanced_copa_system_accuracy (%),openbookqa_icl_total,openbookqa_icl_correct,openbookqa_icl_accuracy (%),openbookqa_system_total,openbookqa_system_correct,openbookqa_system_accuracy (%)
zero_shot_output,1000,804,80.40,1000,811,81.10,1000,898,89.80,1000,906,90.60,1000,852,85.20,1000,861,86.10
icl_random_output,1000,826,82.60,1000,830,83.00,1000,921,92.10,1000,930,93.00,1000,877,87.70,1000,879,87.90
icl_gold_output,1000,821,82.10,1000,836,83.60,1000,922,92.20,1000,935,93.50,1000,873,87.30,1000,880,88.00"""

df = pd.read_csv(StringIO(csv_data))

datasets = ['CommonsenseQA', 'balanced_copa', 'openbookqa']
models = ['zero_shot_output', 'icl_random_output', 'icl_gold_output']

model_names = {
    'zero_shot_output': 'Zero-Shot',
    'icl_random_output': 'ICL Random',
    'icl_gold_output': 'ICL Gold'
}

dataset_names = {
    'CommonsenseQA': 'CommonsenseQA',
    'balanced_copa': 'Balanced COPA',
    'openbookqa': 'OpenBookQA'
}

icl_data = {}
system_data = {}

for dataset in datasets:
    icl_data[dataset] = [df.loc[df['model'] == model, f'{dataset}_icl_accuracy (%)'].values[0] for model in models]
    system_data[dataset] = [df.loc[df['model'] == model, f'{dataset}_system_accuracy (%)'].values[0] for model in models]

improvements = {}
for dataset in datasets:
    improvements[dataset] = [system_data[dataset][i] - icl_data[dataset][i] for i in range(len(models))]

fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=False)

bar_width = 0.35
positions = np.arange(len(models))

colors = ['#3498DB', '#f58676', '#F39C12']
hatches = ['', '//']  # user = no hatch, system = hatch

for i, dataset in enumerate(datasets):
    ax = axes[i]
    
    # bar graph
    icl_bars = ax.bar(positions - bar_width/2, icl_data[dataset], 
                      bar_width, label='User Prompt', 
                      color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
    
    system_bars = ax.bar(positions + bar_width/2, system_data[dataset], 
                         bar_width, label='System Prompt', 
                         color=colors, alpha=0.9, edgecolor='black', linewidth=0.5,
                         hatch='//')
    
    for j, bars in enumerate([icl_bars, system_bars]):
        values = icl_data[dataset] if j == 0 else system_data[dataset]
        for k, bar in enumerate(bars):
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(positions)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    ax.set_xticklabels([model_names[model] for model in models])
    
    ax.set_title(dataset_names[dataset], fontsize=14)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
    min_value = min(min(icl_data[dataset]), min(system_data[dataset])) - 1
    max_value = max(max(icl_data[dataset]), max(system_data[dataset])) + 1
    ax.set_ylim(min_value, max_value)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    if i == 0:
        ax.set_ylabel('Accuracy (\%)', fontsize=12)

max_improvements = {}
for dataset in datasets:
    max_idx = np.argmax(improvements[dataset])
    max_improvements[dataset] = (models[max_idx], improvements[dataset][max_idx])

overall_max = max([(dataset, model, imp) for dataset, (model, imp) in max_improvements.items()], 
                 key=lambda x: x[2])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.08))

plt.savefig('figures/system_ablation.png', dpi=300, bbox_inches='tight')
