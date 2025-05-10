import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import scienceplots
plt.style.use(['science', 'ieee'])

k_values = [1, 2, 4, 8, 16, 32, 64]
methods = ['no_labels', 'random_labels', 'incorrect_labels', 'similar_labels', 'dissimilar_labels']

# Main data
accuracies = {
    'no_labels': [48.70, 48.70, 48.70, 48.70, 48.70, 48.70, 48.70],
    'random_labels': [45.00, 46.80, 50.80, 52.00, 54.10, 54.70, 52.50],
    'incorrect_labels': [43.90, 46.70, 50.50, 51.50, 53.50, 54.80, 53.30],
    'similar_labels': [45.30, 48.80, 50.20, 51.50, 54.60, 53.40, 52.30],
    'dissimilar_labels': [44.20, 48.10, 51.20, 51.60, 54.20, 53.50, 52.90]
}

fig, ax = plt.subplots(figsize=(6, 4))

line_styles = {
    'no_labels': '-',
    'random_labels': '--',
    'incorrect_labels': '-.',
    'similar_labels': ':',
    'dissimilar_labels': '-'
}

markers = {
    'no_labels': 'o',
    'random_labels': 's',
    'incorrect_labels': '^',
    'similar_labels': 'D',
    'dissimilar_labels': 'x'
}

colors = {
    'no_labels': '#444444',
    'random_labels': '#ffce73',
    'incorrect_labels': '#f58676',
    'similar_labels': '#4EAA73',
    'dissimilar_labels': '#E38DBC'
}

method_labels = {
    'no_labels': 'No Examples',
    'random_labels': 'Correct Examples',
    'incorrect_labels': 'Incorrect Examples',
    'similar_labels': 'Similar Examples',
    'dissimilar_labels': 'Dissimilar Examples'
}

for method in methods:
    ax.plot(k_values, accuracies[method], 
            label=method_labels[method], 
            linestyle=line_styles[method],
            marker=markers[method],
            linewidth=1.5,
            color=colors[method],
            markersize=6)

ax.set_xscale('log', base=2)
ax.set_xticks(k_values)
ax.set_xticklabels([str(k) for k in k_values])

ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))
ax.set_ylim(43, 56)

ax.set_xlabel('Number of Examples (k)', fontsize=14)
ax.set_ylabel('Accuracy (\%)', fontsize=14)

ax.grid(linestyle='--', alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.legend(loc='lower right', fontsize=12)

max_method = max(methods, key=lambda m: max(accuracies[m]))
max_k_index = np.argmax(accuracies[max_method])
max_k = k_values[max_k_index]
max_acc = max(accuracies[max_method])

plt.tight_layout()
plt.savefig('figures/qwen_csqa.png', dpi=300, bbox_inches='tight')
