import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from save_svg import save_svg
from scipy import stats
import numpy as np

simtype = 'uncertain_task'
pref = f'../data/{simtype}/noise_simulations/vector_noise_distributed/'
inputs = ['both','vector_only','visual_only']
noise = 0.0
noise_folder = os.path.join(pref, f"noise_{noise:.1f}")

dfs = []
for inp in inputs:
    file_path = os.path.join(noise_folder, f"classified_fields_{inp}.csv")
    dfs.append(pd.read_csv(file_path))
    
def compute_counts(df, name):
    counts = df.groupby(['run', 'unit_type']).size().reset_index(name='count')
    counts['dataset'] = name
    return counts

avg_both = compute_counts(dfs[0], 'both')
avg_visual = compute_counts(dfs[1], 'vector only')
avg_vector = compute_counts(dfs[2], 'visual only')

df_all = pd.concat([avg_both, avg_visual, avg_vector])

# Statistical tests for Place units
# Extract Place unit counts for each condition
place_both = avg_both[avg_both['unit_type'] == 'Place']['count'].values
place_vector_only = avg_visual[avg_visual['unit_type'] == 'Place']['count'].values
place_visual_only = avg_vector[avg_vector['unit_type'] == 'Place']['count'].values

# Store p-values for plotting
p_values = {}

# Independent samples t-test (both vs vector_only)
if len(place_both) > 0 and len(place_vector_only) > 0:
    stat1, p_val1 = stats.ttest_ind(place_both, place_vector_only)
    p_values['both_vs_vector'] = p_val1
    print(f"\n{'='*60}")
    print("Statistical Test: Place units (both vs vector_only)")
    print(f"{'='*60}")
    print(f"Both condition - Mean: {np.mean(place_both):.2f}, Std: {np.std(place_both):.2f}, N: {len(place_both)}")
    print(f"Vector only - Mean: {np.mean(place_vector_only):.2f}, Std: {np.std(place_vector_only):.2f}, N: {len(place_vector_only)}")
    print(f"t-statistic: {stat1:.4f}")
    print(f"P-value: {p_val1:.4f}")
    if p_val1 < 0.05:
        print("Result: Significant difference (p < 0.05)")
    else:
        print("Result: No significant difference (p >= 0.05)")

# Independent samples t-test (both vs visual_only)
if len(place_both) > 0 and len(place_visual_only) > 0:
    stat2, p_val2 = stats.ttest_ind(place_both, place_visual_only)
    p_values['both_vs_visual'] = p_val2
    print(f"\n{'='*60}")
    print("Statistical Test: Place units (both vs visual_only)")
    print(f"{'='*60}")
    print(f"Both condition - Mean: {np.mean(place_both):.2f}, Std: {np.std(place_both):.2f}, N: {len(place_both)}")
    print(f"Visual only - Mean: {np.mean(place_visual_only):.2f}, Std: {np.std(place_visual_only):.2f}, N: {len(place_visual_only)}")
    print(f"t-statistic: {stat2:.4f}")
    print(f"P-value: {p_val2:.4f}")
    if p_val2 < 0.05:
        print("Result: Significant difference (p < 0.05)")
    else:
        print("Result: No significant difference (p >= 0.05)")
    print(f"{'='*60}\n")


#sns.set_context("talk")
colors = sns.color_palette(['#a47ba4ff','#ee9787ff','#e6b04cff','#3b6273ff','#707698ff','#d6819fff'])
#plt.figure(figsize=(8, 6))
ax = sns.barplot(
    data=df_all,
    x='dataset', y='count', hue='unit_type',
    palette=colors,
    errorbar='se',           # show standard deviation
    capsize=0.1,       # small caps on error bars
    err_kws={'linewidth': 0.5}       # thicker error lines
)
sns.despine()
ax.set_ylim(0, 35)
ax.set_ylabel('number of units')
ax.set_xlabel(' ')

# Add significance markers for Place units
def get_significance_marker(p_value):
    """Convert p-value to significance stars"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

# Find the position and height of Place bars for each condition
# Get unique unit types to find Place index
unit_types = df_all['unit_type'].unique()
place_idx = np.where(unit_types == 'Place')[0]

if len(place_idx) > 0:
    place_idx = place_idx[0]
    
    # Get the bar positions for each dataset
    datasets = ['both', 'vector only', 'visual only']
    
    # Calculate mean heights for Place units in each condition
    place_heights = {}
    for i, dataset in enumerate(datasets):
        data_subset = df_all[(df_all['dataset'] == dataset) & (df_all['unit_type'] == 'Place')]
        if len(data_subset) > 0:
            place_heights[dataset] = data_subset['count'].mean()
    
    # Get bar width from the plot
    bar_width = ax.patches[0].get_width()
    n_datasets = len(datasets)
    n_unit_types = len(unit_types)
    
    # Calculate x positions for each bar group
    group_width = bar_width * n_unit_types
    
    # Add significance markers
    y_max = ax.get_ylim()[1]
    line_height = y_max * 0.85  # Position for significance lines
    
    # Both vs Vector only
    if 'both_vs_vector' in p_values:
        sig_marker = get_significance_marker(p_values['both_vs_vector'])
        if sig_marker != 'ns':
            x1 = 0  # 'both' position
            x2 = 1  # 'vector only' position
            
            # Draw line
            ax.plot([x1, x1, x2, x2], 
                   [line_height, line_height + 0.5, line_height + 0.5, line_height], 
                   'k-', linewidth=0.5)
            # Add star
            ax.text((x1 + x2) / 2, line_height + 0.8, sig_marker, 
                   ha='center', va='bottom', fontsize=8)
    
    # Both vs Visual only
    if 'both_vs_visual' in p_values:
        sig_marker = get_significance_marker(p_values['both_vs_visual'])
        if sig_marker != 'ns':
            x1 = 0  # 'both' position
            x2 = 2  # 'visual only' position
            
            # Draw line at a higher position
            line_height2 = y_max * 0.92
            ax.plot([x1, x1, x2, x2], 
                   [line_height2, line_height2 + 0.5, line_height2 + 0.5, line_height2], 
                   'k-', linewidth=0.5)
            # Add star
            ax.text((x1 + x2) / 2, line_height2 + 0.8, sig_marker, 
                   ha='center', va='bottom', fontsize=8)
#plt.legend(title='Unit Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Continuous inputs")
#plt.tight_layout()
plt.show()

save_svg(ax, "figs/summary_continuous.svg", 45, 35, font_size=6, line_scale=0.4)