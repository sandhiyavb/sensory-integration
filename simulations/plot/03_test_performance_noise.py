#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import spearmanr

simtype='continuous'

def test_direct_finding_rate(test_data_csv, path_length=1.3*8) : 
    test_data = pd.read_csv(test_data_csv, sep='\t')
    filtered_data = test_data[test_data['steps'] < path_length]
    count = filtered_data.groupby('run').size()    
    n_trials = test_data.groupby('run').size().mean()
    mean_count = count.mean()
    std_count = count.std()
        
    return mean_count/n_trials, std_count/n_trials

def test_direct_finding_per_run(test_data_csv, path_length=1.3*8):
    test_data = pd.read_csv(test_data_csv, sep='\t')
    
    filtered_data = test_data[test_data['steps'] < path_length]
    
    counts = filtered_data.groupby('run').size()
    n_trials = test_data.groupby('run').size()
    
    counts = counts.reindex(n_trials.index, fill_value=0)
    
    rates = counts / n_trials
    
    return rates.values

def plot_test_bars(data_sources, spacing=0, label=None, ax=None, color='grey',
                   input_types = ['Visual only', 'Vector only', 'Both']) : 
    means = []
    stds   = []
    
    for d in data_sources : 
        mean, std = test_direct_finding_rate(d, 1.3*8)
        means.append(mean)
        stds.append(std)
    
    # Create lists for the plot
    stds = [np.zeros(len(stds)), stds]
    x_pos = np.arange(len(input_types))
    
    # Build the plot    
    ax.bar(x_pos+spacing, means, yerr=stds, align='center',width=0.2, alpha=0.9, ecolor='#333333', linewidth=0.5,
           capsize=1.5, color=color,label=label, edgecolor = "gray")
    
    ax.set_ylabel('Proportion successful test trials')
    ax.set_xlabel('Noise')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(input_types)
    ax.set_title('Test performance')
    ax.set_ylim(top=1.2)
    ax.yaxis.grid(True)
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7,7), dpi=300)

#Comparison of test performance
prefs = [f'../data/{simtype}/noise_simulations/vector_noise_position/',
         f'../data/{simtype}/noise_simulations/visual_noise_position/',
         f'../data/{simtype}/noise_simulations/vector_noise_distributed/',
         f'../data/{simtype}/noise_simulations/visual_noise_distributed/']
axs = axs.ravel()
random_agent = '../data/test_log_random_agent.csv'
test_data = pd.read_csv(random_agent, sep='\t')
test_data = test_data.apply(pd.to_numeric, errors='coerce')
test_data = test_data.dropna()
test_data.to_csv(random_agent, sep='\t', encoding='utf-8',index=False)
random_mean, std_mean = test_direct_finding_rate(random_agent)

for i in range(len(prefs)) : 
    pref = prefs[i]
    folders = [f for f in os.listdir(pref)]
    folders.sort()
    
    
    noises = np.arange(0.0,0.6,0.1)
    test_logs = ['test_log_vector_only.csv', 'test_log_visual_only.csv',
                 'test_log_both.csv']
    
    test_data = [[pref+f+'/'+test for test in test_logs] for f in folders]
    labels = ["%.1f"%noise for noise in noises]
        
    vector_only = [t[0] for t  in test_data]
    visual_only = [t[1] for t  in test_data]
    both = [t[2] for t  in test_data]
    
    mean_vector_only = []
    sem_vector_only = []
    mean_visual_only = []
    sem_visual_only = []
    mean_both = []
    sem_both = []
    
    for v in vector_only : 
        rates = test_direct_finding_per_run(v)
        mean_vector_only.append(np.mean(rates))
        sem_vector_only.append(np.std(rates, ddof=1) / np.sqrt(len(rates)))
        
    for v in visual_only : 
        rates = test_direct_finding_per_run(v)
        mean_visual_only.append(np.mean(rates))
        sem_visual_only.append(np.std(rates, ddof=1) / np.sqrt(len(rates)))
        
    for v in both : 
        rates = test_direct_finding_per_run(v)
        mean_both.append(np.mean(rates))
        sem_both.append(np.std(rates, ddof=1) / np.sqrt(len(rates)))
    

    
    axs[i].errorbar(noises, mean_vector_only, yerr=sem_vector_only,
                marker='o', color='#454545', label='vector only',
                linewidth=3, markersize=6, capsize=3)

    axs[i].errorbar(noises, mean_visual_only, yerr=sem_visual_only,
                marker='o', color='#E3B23C', label='visual only',
                linewidth=3, markersize=6, capsize=3)

    axs[i].errorbar(noises, mean_both, yerr=sem_both,
                marker='o', color='#BFAF80', label='both',
                linewidth=3, markersize=6, capsize=3)
    #axs[i].legend_.remove()
    axs[i].set_xticks(noises)
    line = axs[i].axhline(y=random_mean, color='black', linestyle='--', linewidth=1, label='random agent')

fig.text(0.55, 0.00, "Noise", ha="center", fontsize=12)   # x-axis
fig.text(0.06, 0.55, "Proportion successful test trials", va="center", rotation="vertical", fontsize=12)  # y-axis

col_labels = ["Trained with vector noise", "Trained with visual noise"]
row_labels = ["Pose noise", "Sensor noise"]

for ax, col in zip(axs[:2], col_labels):  # top row
    ax.set_title(col, fontsize=14)

for ax, row in zip(axs[::2], row_labels):  # left column
    ax.set_ylabel(row, fontsize=14, labelpad=20)
    
handles, labels = axs[1].get_legend_handles_labels()
#sns.despine()
sns.set_style("white")
sns.set_style("ticks")
plt.tight_layout()
plt.rcParams.update({'font.size': 12})
plt.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.0, -0.35), shadow=True, ncols=4)

if not os.path.exists('figs') :
    os.makedirs('figs')
fig.savefig(f'figs/{simtype}_noise_types.svg', format='svg', dpi=300)


# STATISTICAL ANALYSIS





# Organize data by condition
# Structure: test_data_organized[condition][noise_type][input_type][noise_level]
test_data_organized = {
    'vector_noise_position': {},
    'visual_noise_position': {},
    'vector_noise_distributed': {},
    'visual_noise_distributed': {}
}

condition_names = ['vector_noise_position', 'visual_noise_position', 
                   'vector_noise_distributed', 'visual_noise_distributed']

for i, pref in enumerate(prefs):
    cond_name = condition_names[i]
    folders = sorted([f for f in os.listdir(pref)])
    
    noises_list = np.arange(0.0, 0.6, 0.1)
    test_logs = ['test_log_vector_only.csv', 'test_log_visual_only.csv', 'test_log_both.csv']
    input_type_names = ['vector_only', 'visual_only', 'both']
    
    for noise_level, folder in zip(noises_list, folders):
        noise_key = f"{noise_level:.1f}"
        if noise_key not in test_data_organized[cond_name]:
            test_data_organized[cond_name][noise_key] = {}
        
        for input_type, test_log in zip(input_type_names, test_logs):
            file_path = pref + folder + '/' + test_log
            rates = test_direct_finding_per_run(file_path)
            test_data_organized[cond_name][noise_key][input_type] = rates


print("\n1. NOISE LEVEL EFFECT: POSE vs. SENSOR (by input type)")

# Include all noise levels
noises_all = [f"{n:.1f}" for n in np.arange(0.0, 0.6, 0.1)]

print("\n1a. COMBINED (vector-trained + visual-trained):")
print("-" * 70)

for input_type in ['vector_only', 'visual_only', 'both']:
    print(f"\n{input_type.upper().replace('_', ' ')}:")
    
    # Combine pose noise conditions (vector_noise_position + visual_noise_position)
    pose_performance = []
    pose_noise_levels = []
    for noise in noises_all:
        # Get all runs from both pose conditions
        vector_pos_rates = test_data_organized['vector_noise_position'][noise][input_type]
        visual_pos_rates = test_data_organized['visual_noise_position'][noise][input_type]
        combined = np.concatenate([vector_pos_rates, visual_pos_rates])
        pose_performance.extend(combined)
        pose_noise_levels.extend([float(noise)] * len(combined))
    
    # Combine sensor noise conditions (vector_noise_distributed + visual_noise_distributed)
    sensor_performance = []
    sensor_noise_levels = []
    for noise in noises_all:
        vector_dist_rates = test_data_organized['vector_noise_distributed'][noise][input_type]
        visual_dist_rates = test_data_organized['visual_noise_distributed'][noise][input_type]
        combined = np.concatenate([vector_dist_rates, visual_dist_rates])
        sensor_performance.extend(combined)
        sensor_noise_levels.extend([float(noise)] * len(combined))
    
    # Calculate correlations (noise level vs performance)
    pose_corr, pose_p = spearmanr(pose_noise_levels, pose_performance)
    sensor_corr, sensor_p = spearmanr(sensor_noise_levels, sensor_performance)
    
    print(f"  Pose noise - correlation: {pose_corr:.4f}, p-value: {pose_p:.6f}")
    if pose_p < 0.05:
        print("    *** Significant correlation (p < 0.05)")
    
    print(f"  Sensor noise - correlation: {sensor_corr:.4f}, p-value: {sensor_p:.6f}")
    if sensor_p < 0.05:
        print("    *** Significant correlation (p < 0.05)")
    
    # Compare the strength of correlations
    if abs(pose_corr) > abs(sensor_corr):
        diff = abs(pose_corr) - abs(sensor_corr)
        print(f"   Pose noise shows stronger correlation (|r| difference = {diff:.4f})")
    else:
        diff = abs(sensor_corr) - abs(pose_corr)
        print(f"   Sensor noise shows stronger correlation (|r| difference = {diff:.4f})")

# Test separately for vector-trained and visual-trained conditions
print("\n\n1b. VECTOR-TRAINED ONLY (vector_noise_position vs. vector_noise_distributed):")


for input_type in ['vector_only', 'visual_only', 'both']:
    print(f"\n{input_type.upper().replace('_', ' ')}:")
    
    # Pose noise: vector_noise_position
    pose_performance = []
    pose_noise_levels = []
    for noise in noises_all:
        rates = test_data_organized['vector_noise_position'][noise][input_type]
        pose_performance.extend(rates)
        pose_noise_levels.extend([float(noise)] * len(rates))
    
    # Sensor noise: vector_noise_distributed
    sensor_performance = []
    sensor_noise_levels = []
    for noise in noises_all:
        rates = test_data_organized['vector_noise_distributed'][noise][input_type]
        sensor_performance.extend(rates)
        sensor_noise_levels.extend([float(noise)] * len(rates))
    
    # Calculate correlations
    pose_corr, pose_p = spearmanr(pose_noise_levels, pose_performance)
    sensor_corr, sensor_p = spearmanr(sensor_noise_levels, sensor_performance)
    
    print(f"  Pose noise - correlation: {pose_corr:.4f}, p-value: {pose_p:.6f}")
    if pose_p < 0.05:
        print("    *** Significant correlation (p < 0.05)")
    
    print(f"  Sensor noise - correlation: {sensor_corr:.4f}, p-value: {sensor_p:.6f}")
    if sensor_p < 0.05:
        print("    *** Significant correlation (p < 0.05)")
    
    if abs(pose_corr) > abs(sensor_corr):
        diff = abs(pose_corr) - abs(sensor_corr)
        print("   Pose noise shows stronger correlation (|r| difference = {diff:.4f})")
    else:
        diff = abs(sensor_corr) - abs(pose_corr)
        print("   Sensor noise shows stronger correlation (|r| difference = {diff:.4f})")

print("\n\n1c. VISUAL-TRAINED ONLY (visual_noise_position vs. visual_noise_distributed):")
print("-" * 70)

for input_type in ['vector_only', 'visual_only', 'both']:
    print(f"\n{input_type.upper().replace('_', ' ')}:")
    
    # Pose noise: visual_noise_position
    pose_performance = []
    pose_noise_levels = []
    for noise in noises_all:
        rates = test_data_organized['visual_noise_position'][noise][input_type]
        pose_performance.extend(rates)
        pose_noise_levels.extend([float(noise)] * len(rates))
    
    # Sensor noise: visual_noise_distributed
    sensor_performance = []
    sensor_noise_levels = []
    for noise in noises_all:
        rates = test_data_organized['visual_noise_distributed'][noise][input_type]
        sensor_performance.extend(rates)
        sensor_noise_levels.extend([float(noise)] * len(rates))
    
    # Calculate correlations
    pose_corr, pose_p = spearmanr(pose_noise_levels, pose_performance)
    sensor_corr, sensor_p = spearmanr(sensor_noise_levels, sensor_performance)
    
    print(f"  Pose noise - correlation: {pose_corr:.4f}, p-value: {pose_p:.6f}")
    if pose_p < 0.05:
        print("    *** Significant correlation (p < 0.05)")
    
    print(f"  Sensor noise - correlation: {sensor_corr:.4f}, p-value: {sensor_p:.6f}")
    if sensor_p < 0.05:
        print("    *** Significant correlation (p < 0.05)")
    
    if abs(pose_corr) > abs(sensor_corr):
        diff = abs(pose_corr) - abs(sensor_corr)
        print("  Pose noise shows stronger correlation (|r| difference = {diff:.4f})")
    else:
        diff = abs(sensor_corr) - abs(pose_corr)
        print("  Sensor noise shows stronger correlation (|r| difference = {diff:.4f})")


# 1d. Specific test: Visual sensor noise, visual_only, noise 0.0 vs 0.2

print("\n\n1d. SPECIFIC TEST: Visual sensor noise, visual_only, noise 0.0 vs. 0.2")


noise_0_rates = test_data_organized['visual_noise_distributed']['0.0']['vector_only']
noise_01_rates = test_data_organized['visual_noise_distributed']['0.2']['vector_only']

t_stat, p_val = ttest_ind(noise_0_rates, noise_01_rates)
print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.6f}")
print(f"  Noise 0.0 mean: {np.mean(noise_0_rates):.4f}")
print(f"  Noise 0.1 mean: {np.mean(noise_01_rates):.4f}")
if p_val < 0.05:
    print("  *** Significant difference (p < 0.05)")
else:
    print("  No significant difference")


# 2. Performance difference for noisy signals: pose vs sensor
#    Compare vector_only (for vector-trained) and visual_only (for visual-trained)

print("\n\n2. NOISY SIGNAL PERFORMANCE: POSE vs. SENSOR")




# Vector-trained agents tested with vector_only input
print("\nVector-trained agents (vector_only test):")
vector_pose_performance = np.concatenate([
    test_data_organized['vector_noise_position'][noise]['vector_only'] 
    for noise in noises_all
])
vector_sensor_performance = np.concatenate([
    test_data_organized['vector_noise_distributed'][noise]['vector_only'] 
    for noise in noises_all
])

t_stat, p_val = ttest_ind(vector_pose_performance, vector_sensor_performance)
print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.6f}")
print(f"  Pose noise mean: {np.mean(vector_pose_performance):.4f}")
print(f"  Sensor noise mean: {np.mean(vector_sensor_performance):.4f}")
if p_val < 0.05:
    print("  *** Significant difference (p < 0.05)")
else:
    print("  No significant difference")

# Visual-trained agents tested with visual_only input
print("\nVisual-trained agents (visual_only test):")
visual_pose_performance = np.concatenate([
    test_data_organized['visual_noise_position'][noise]['visual_only'] 
    for noise in noises_all
])
visual_sensor_performance = np.concatenate([
    test_data_organized['visual_noise_distributed'][noise]['visual_only'] 
    for noise in noises_all
])

t_stat, p_val = ttest_ind(visual_pose_performance, visual_sensor_performance)
print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.6f}")
print(f"  Pose noise mean: {np.mean(visual_pose_performance):.4f}")
print(f"  Sensor noise mean: {np.mean(visual_sensor_performance):.4f}")
if p_val < 0.05:
    print("  *** Significant difference (p < 0.05)")
else:
    print("  No significant difference")
