import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

noises = np.arange(0.0,0.6,0.1)
inputs = ['both','vector_only','visual_only']
simtype = 'certain_task'
prefs = [f'../data/{simtype}/noise_simulations/vector_noise_position/',
         f'../data/{simtype}/noise_simulations/visual_noise_position/',
         f'../data/{simtype}/noise_simulations/vector_noise_distributed/',
         f'../data/{simtype}/noise_simulations/visual_noise_distributed/']

unit_type = "Place"

results_mean = {pref: {inp: [] for inp in inputs} for pref in prefs}
results_sem  = {pref: {inp: [] for inp in inputs} for pref in prefs}

for pref in prefs:
    for noise in noises:
        noise_folder = os.path.join(pref, f"noise_{noise:.1f}")
        for inp in inputs:
            file_path = os.path.join(noise_folder, f"classified_fields_{inp}.csv")
            if not os.path.exists(file_path):
                print(f"Missing: {file_path}")
                results_mean[pref][inp].append(np.nan)
                results_sem[pref][inp].append(np.nan)
                continue

            df = pd.read_csv(file_path)

            # Filter rows of interest
            df_place = df[df['unit_type'] == unit_type]

            # Count occurrences per run
            counts_per_run = df_place.groupby("run").size()

            # Average across runs
            if not counts_per_run.empty:
                mean = counts_per_run.mean()
                sem = counts_per_run.std(ddof=1) / np.sqrt(len(counts_per_run))
            else:
                mean = 0
                sem = 0
            
            results_mean[pref][inp].append(mean)
            results_sem[pref][inp].append(sem)


fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True, dpi=300)
axes = axes.ravel()
colors = {'both': '#BFAF80',
          'vector_only' : '#454545',
              'visual_only' : '#E3B23C'}
for i, pref in enumerate(prefs):
    ax = axes[i]
    for inp in inputs:        
        ax.errorbar(noises,
                    results_mean[pref][inp],
                    yerr=results_sem[pref][inp],
                    marker='o',
                    color=colors[inp],
                    label=inp,
                    linewidth=3,
                    markersize=6,
                    capsize=3)
        #ax.legend_.remove()
        ax.set_xticks(noises)

    ax.set_xlabel("Noise")

fig.text(0.06, 0.55, f"#{unit_type} cells", va="center", rotation="vertical", fontsize=12)  # y-axis

# ---- Add column and row labels ----
col_labels = ["Trained with vector noise", "Trained with visual noise"]
row_labels = ["Position noise", "Distributed noise"]

for ax, col in zip(axes[:2], col_labels):  # top row
    ax.set_title(col, fontsize=14)

for ax, row in zip(axes[::2], row_labels):  # left column
    ax.set_ylabel(row, fontsize=14, labelpad=20)
    
handles, labels = axes[1].get_legend_handles_labels()
sns.despine()
sns.set_style("white")
sns.set_style("ticks")
plt.tight_layout()
plt.rcParams.update({'font.size': 14})
plt.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.0, -0.35), shadow=True, ncols=4)
fig.savefig('figs/place_fields_noise.svg', format='svg', dpi=300)

