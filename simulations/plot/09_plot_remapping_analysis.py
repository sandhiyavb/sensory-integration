
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import euclidean

from save_svg import save_svg

#Check average shift in place cell center in the two task conditions

simtypes = ['uncertain_task', 'certain_task']
inputs = ['both','vector_only','visual_only']
noise = 0.0

all_results = []  
threshold = 0.7
for simtype in simtypes:
    pref = f'../data/{simtype}/noise_simulations/vector_noise_distributed/'
    noise_folder = os.path.join(pref, f"noise_{noise:.1f}")


    dfs = []
    for inp in inputs:
        file_path = os.path.join(noise_folder, f"classified_fields_{inp}.csv")
        dfs.append(pd.read_csv(file_path))

    df_both = dfs[0][dfs[0]['unit_type']=='Place'].drop('unit_type',axis=1) 
    df_vector_only = dfs[1][dfs[1]['unit_type']=='Place'].drop('unit_type',axis=1) 
    df_visual_only = dfs[2][dfs[2]['unit_type']=='Place'].drop('unit_type',axis=1) 

    #Common units
    runs_both_vector = set(df_both['run']).intersection(df_vector_only['run'])
    runs_both_visual = set(df_both['run']).intersection(df_visual_only['run'])

    common_both_vector = {}
    for run in runs_both_vector:
        units_both = set(df_both.loc[df_both['run'] == run, 'unit'])
        units_vector = set(df_vector_only.loc[df_vector_only['run'] == run, 'unit'])
        common_both_vector[run] = units_both & units_vector

    common_both_visual = {}
    for run in runs_both_visual:
        units_both = set(df_both.loc[df_both['run'] == run, 'unit'])
        units_visual = set(df_visual_only.loc[df_visual_only['run'] == run, 'unit'])
        common_both_visual[run] = units_both & units_visual


    run_stats = []
    for run in runs_both_vector:
        act_both = np.load(os.path.join(noise_folder, f'run_{run}', 'activations', 'activations_both.npy'))
        act_vec  = np.load(os.path.join(noise_folder, f'run_{run}', 'activations', 'activations_vector_only.npy'))
        act_vis  = np.load(os.path.join(noise_folder, f'run_{run}', 'activations', 'activations_visual_only.npy'))
        
        act_both = np.squeeze(act_both)
        act_vec  = np.squeeze(act_vec)
        act_vis  = np.squeeze(act_vis)

        #VECTOR comparison
        units_vec = list(common_both_vector.get(run, []))
        dists_vec = []
        stationary_vec = []
        
        for unit in units_vec:
            both_map = np.reshape(np.mean(act_both[:, :, unit],axis=0), (25,25))
            vec_map  = np.reshape(np.mean(act_vec[:, :, unit],axis=0), (25,25))
            dist = euclidean(center_of_mass(both_map), center_of_mass(vec_map))
            if dist > threshold : 
                dists_vec.append(dist)
            else : 
                stationary_vec.append(dist)

        #VISUAL comparison
        units_vis = list(common_both_visual.get(run, []))
        dists_vis = []
        stationary_vis = []
        for unit in units_vis:
            both_map = np.reshape(np.mean(act_both[:, :, unit], axis=0), (25,25))
            vis_map  = np.reshape(np.mean(act_vis[:, :, unit], axis=0), (25,25))
            dist = euclidean(center_of_mass(both_map), center_of_mass(vis_map))
            if dist > threshold : 
                dists_vis.append(dist)
            else :
                stationary_vis.append(dist)

        run_stats.append({
            'simtype': simtype,
            'run': run,
            'mean_vector': np.mean(dists_vec) if dists_vec else np.nan,
            'std_vector': np.std(dists_vec) if dists_vec else np.nan,
            'mean_visual': np.mean(dists_vis) if dists_vis else np.nan,
            'std_visual': np.std(dists_vis) if dists_vis else np.nan,
            'stat_vector': len(stationary_vec)/len(units_vec) if units_vec else 0.0,
            'n_units_vec' : len(units_vec),
            'stat_visual' : len(stationary_vis)/len(units_vis) if units_vis else 0.0,
            'n_units_vis' : len(units_vis)
        })

    all_results.extend(run_stats)

results_df = pd.DataFrame(all_results)
print(results_df.head())

plot_df = results_df.melt(
    id_vars=['simtype', 'run'],
    value_vars=['mean_vector', 'mean_visual'],
    var_name='condition',
    value_name='mean_distance'
)
plot_df['condition'] = plot_df['condition'].map({
    'mean_vector': 'Vector only',
    'mean_visual': 'Visual only'
})

label_map = {
    'uncertain_task': 'Intermittent',
    'certain_task': 'Continuous'
}
plot_df['simtype_label'] = plot_df['simtype'].map(label_map)

plot_df_2 = results_df.melt(
    id_vars=['simtype', 'run'],
    value_vars=['stat_vector', 'stat_visual'],
    var_name='condition',
    value_name='prop_stat_fields'
)
plot_df_2['condition'] = plot_df_2['condition'].map({
    'stat_vector': 'Vector only',
    'stat_visual': 'Visual only'
})

plot_df_2['simtype_label'] = plot_df['simtype'].map(label_map)


fig, ax = plt.subplots()
ax = sns.barplot(
    data=plot_df_2,
    x='simtype_label', y='prop_stat_fields', hue='condition',
    errorbar='se',  
    palette='Set2',
    capsize=0.1
)

ax.set_xlabel(' ')
ax.set_ylabel('Proportion of place cells', fontsize=15)

ax.tick_params(labelsize=15)
plt.show()
fig.savefig('figs/stationary_fields.svg', format='svg', dpi=300)