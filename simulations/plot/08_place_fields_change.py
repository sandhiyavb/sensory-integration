
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from save_svg import save_svg

#Plot what the place fields in the both condition change to with only vector/visual inputs
simtypes = ['uncertain_task', 'certain_task']
task = simtypes[0] #Set which task condition to look at
inputs = ['both','vector_only','visual_only']
noise = 0.0

all_results = []  

for simtype in simtypes:
    pref = f'../data/{simtype}/noise_simulations/visual_noise_distributed/' #with zero noise doesn't matter which one
    noise_folder = os.path.join(pref, f"noise_{noise:.1f}")


    dfs = []
    
    for inp in inputs:
        file_path = os.path.join(noise_folder, f"classified_fields_{inp}.csv")
        dfs.append(pd.read_csv(file_path))

    place_units = dfs[0][dfs[0]['unit_type']=='Place'].groupby('run')['unit']
    df_both = dfs[0][dfs[0]['unit_type']=='Place']
    df_vector_only  = dfs[1].merge(df_both[['run', 'unit']], on=['run', 'unit'], how='inner').drop(['trial','unit'],axis=1)
    df_visual_only = dfs[2].merge(df_both[['run', 'unit']], on=['run', 'unit'], how='inner').drop(['trial','unit'],axis=1)

    counts_vector = df_vector_only['unit_type'].value_counts(normalize=True)
    counts_visual = df_visual_only['unit_type'].value_counts(normalize=True)
    
    run_counts_vector = (
    df_vector_only.groupby('run')['unit_type']
    .value_counts(normalize=True)
    .rename('prop')
    .reset_index()
)
    
    avg_prop_vector = run_counts_vector.groupby('unit_type')['prop'].mean().reset_index()
    avg_prop_vector['task'] = simtype
    avg_prop_vector['input'] = 'vector only'
    all_results.append(avg_prop_vector)
    
    run_counts_visual = (
    df_visual_only.groupby('run')['unit_type']
    .value_counts(normalize=True)
    .rename('prop')
    .reset_index()
)
    
    avg_prop_visual = run_counts_visual.groupby('unit_type')['prop'].mean().reset_index()
    avg_prop_visual['task'] = simtype
    avg_prop_visual['input'] = 'visual only'
    all_results.append(avg_prop_visual)


final_df = pd.concat(all_results, ignore_index=True)   
pivot_table = final_df.pivot_table(values='prop', index=['task', 'input'], 
                                   columns='unit_type', fill_value=0.0)


tasks = sorted(final_df['task'].unique())
inputs = sorted(final_df['input'].unique())
unit_types = sorted(final_df['unit_type'].unique())

colors = sns.color_palette(['#a47ba4ff','#ee9787ff','#e6b04cff','#3b6273ff','#707698ff','#d6819fff'])
#sns.set_context('poster')
fig, ax = plt.subplots()

subset = final_df[final_df['task'] == task]

pivot_df = (
    subset.pivot_table(
        index='input',
        columns='unit_type',
        values='prop',
        fill_value=0
    )
    .reindex(index=inputs, columns=unit_types, fill_value=0)
)

bottom = pd.Series(0, index=pivot_df.index, dtype=float)

for ut, color in zip(unit_types, colors):
    bars = ax.bar(
        pivot_df.index,
        pivot_df[ut],
        bottom=bottom,
        color=color,
        edgecolor='white',
        label=ut
    )



    bottom += pivot_df[ut]

ax.set_title("Intermittent inputs")
ax.set_ylabel("Proportion")
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
sns.despine()
plt.show()

save_svg(ax, "figs/change_continuous.svg", 15, 25, font_size=6, line_scale=0.4)
