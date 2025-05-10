import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import scienceplots
plt.style.use(['science', 'ieee'])

k_values = [1, 4, 16, 32]
methods = ['no_labels', 'random_labels', 'incorrect_labels']

# Main data
accuracies = {
    'no_labels': [85.1] * 4,
    'random_labels': [85.40, 86.43, 87.20, 86.70],
    'incorrect_labels': [85.37, 86.37, 87.47, 86.67]
}

fig, ax = plt.subplots(figsize=(5, 3.7))

line_styles = {
    'no_labels': '-',
    'random_labels': '--',
    'incorrect_labels': '-.'
}

markers = {
    'no_labels': 'o',
    'random_labels': 's',
    'incorrect_labels': '^'
}

colors = {
    'no_labels': '#444444',
    'random_labels': '#ffce73',
    'incorrect_labels': '#f58676'
}

# Legend labels
method_labels = {
    'no_labels': 'No Demos',
    'random_labels': 'Demos w/ gold labels',
    'incorrect_labels': 'Demos w/ random labels'
}

for method in methods:
    ax.plot(k_values, accuracies[method], 
            label=method_labels[method], 
            linestyle=line_styles[method],
            marker=markers[method],
            linewidth=1.5,
            color=colors[method],
            markersize=6)

# log-scale x-axis
ax.set_xscale('log', base=2)
ax.set_xticks(k_values)
ax.set_xticklabels([str(k) for k in k_values])

ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))
ax.set_ylim(84-0.1, 88-0.2)

ax.set_xlabel('Number of Demonstrations ($k$)', fontsize=14)
ax.set_ylabel('Accuracy (\%)', fontsize=14)

ax.grid(linestyle='--', alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.legend(loc='lower right', fontsize=12)

plt.tight_layout()
plt.savefig('figures/base_k_ablate.png', dpi=300, bbox_inches='tight')
