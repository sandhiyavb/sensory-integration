#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import f_oneway


from analysis.learning_and_behavior import (
    smooth_reward,
    mean_std_reward,
    plot_reward
)

def direct_paths_to_goal(data, n_steps=1.3*8, n_trials=300, n_blocks=5) :    
    """
    Calculate number of direct paths taken to the goal in the first N trials.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    n_steps : TYPE, optional
        DESCRIPTION. The default is 25.
    n_trials : TYPE, optional
        DESCRIPTION. The default is 300.
    n_blocks : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    mean_el : TYPE
        DESCRIPTION.
    std_el : TYPE
        DESCRIPTION.

    """
    runs = np.unique(data[:,0])
    paths_to_goal = np.stack([data[np.where(data[:,0]==i)][:,2] for i in runs])[:,:n_trials]
    indices = np.linspace(0, paths_to_goal.shape[1], n_blocks+1, dtype='int32')
    average = []
    for e in paths_to_goal : 
        a = []
        for i in range(len(indices)-1) :
            sim_slice = e[indices[i]:indices[i+1]]
            n_direct_finds = (sim_slice <= n_steps).sum()
            proportion = n_direct_finds/len(sim_slice)
            a.append(proportion)
        average.append(a)
        
    mean_el = np.mean(np.array(average),axis=0)
    std_el = np.std(np.array(average),axis=0)
    
    return mean_el, std_el

def direct_paths_per_run(data, n_steps=1.3*8, n_trials=300, n_blocks=5):
    """Return per-run direct finding proportions per block"""
    
    runs = np.unique(data[:,0])
    paths_to_goal = np.stack([
        data[np.where(data[:,0]==i)][:,2] for i in runs
    ])[:, :n_trials]
    
    indices = np.linspace(0, paths_to_goal.shape[1], n_blocks+1, dtype='int32')
    
    all_runs = []
    for e in paths_to_goal:
        a = []
        for i in range(len(indices)-1):
            sim_slice = e[indices[i]:indices[i+1]]
            n_direct_finds = (sim_slice <= n_steps).sum()
            proportion = n_direct_finds / len(sim_slice)
            a.append(proportion)
        all_runs.append(a)
    
    return np.array(all_runs)  # shape: (runs, blocks)

sns.set_style("white")
sns.set_style("ticks")
plt.rcParams.update({'font.size': 12})

simtype = 'certain_task'
file_name = '/training_log.csv'

base_path = f'../data/{simtype}/noise_simulations/'
prefs = [
    base_path + 'vector_noise_position/',
    base_path + 'visual_noise_position/',
    base_path + 'vector_noise_distributed/',
    base_path + 'visual_noise_distributed/'
]


noises = np.arange(0.0, 0.6, 0.1)
labels = [f"{n:.1f}" for n in noises]
colors = plt.get_cmap('Greys')(np.linspace(0.3, 1.0, len(noises)))


fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7, 7), dpi=300)
fig1, axs1 = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7, 7), dpi=300)
axs, axs1 = axs.ravel(), axs1.ravel()

df_data = {}  
for i, pref in enumerate(prefs):
    folders = sorted(os.listdir(pref))
    condition_name = pref.split('/')[-2]  
    df_data[condition_name] = {}
    
    means_df, stds_df = [], []
    n_trials = 1000

    
    for j, folder in enumerate(folders):
        
        training_data = np.genfromtxt(pref + folder + file_name)[1:]
        
        df_runs = direct_paths_per_run(training_data, n_steps=1.3*8, n_trials=n_trials)
        noise_level = labels[j]
        df_data[condition_name][noise_level] = df_runs
                
        mean_r, std_r = mean_std_reward(training_data, n_trials)
        mean_reward = smooth_reward(mean_r)
        std_reward = smooth_reward(std_r)

        #Learning curves
        plot_reward(mean_reward, axs[i], colors[j], labels[j])

        #Direct finding data
        mean_df, std_df = direct_paths_to_goal(training_data, n_steps=1.3*8, n_trials=n_trials)
        means_df.append(mean_df)
        stds_df.append(std_df)

for cond_idx, cond in enumerate(df_data):
    for j, noise in enumerate(labels):
        vals = df_data[cond][noise]  # shape: (runs, blocks)
        
        mean = np.mean(vals, axis=0)
        sem = np.std(vals, axis=0, ddof=1) / np.sqrt(vals.shape[0])
        
        axs1[cond_idx].errorbar(
            np.arange(len(mean)),
            mean,
            yerr=sem,
            marker='o',
            c=colors[j],
            label=noise
        )
        axs1[i].set_xticks(np.arange(5))
        axs1[i].set_xticklabels(np.arange(1, 6))
        axs1[i].set_ylim(0.0, 1.0)

col_labels = ["Trained with vector noise", "Trained with visual noise"]
row_labels = ["Pose noise", "Sensor noise"]


for ax, col in zip(axs[:2], col_labels):
    ax.set_title(col, fontsize=14)
for ax, row in zip(axs[::2], row_labels):
    ax.set_ylabel(row, fontsize=14, labelpad=20)
fig.text(0.06, 0.55, "Trial reward", va="center", rotation="vertical", fontsize=12)


for ax, col in zip(axs1[:2], col_labels):
    ax.set_title(col, fontsize=14)
for ax, row in zip(axs1[::2], row_labels):
    ax.set_ylabel(row, fontsize=14, labelpad=20)
fig1.text(0.06, 0.55, "Proportion of direct finding", va="center", rotation="vertical", fontsize=12)


handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles=handles,
    labels=labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    shadow=False,
    ncols=6,
    title="Noise Level"
)
sns.despine()
handles1, labels1 = axs1[0].get_legend_handles_labels()
fig1.legend(
    handles=handles1,
    labels=labels1,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    shadow=False,
    ncols=6,
    title="Noise Level"
)
sns.despine()
plt.tight_layout()
os.makedirs('figs', exist_ok=True)

fig.savefig(f'figs/{simtype}_learning_noise.svg', format='svg', dpi=300)
fig1.savefig(f'figs/{simtype}_direct_finding.svg', format='svg', dpi=300)

plt.show()


# STATISTICAL ANALYSES

# Prepare data for analysis
# Combine pose noise conditions (vector and visual)
pose_noise_data = {}  # noise_level -> all runs
for noise in labels:
    pose_noise_data[noise] = np.vstack([
        df_data['vector_noise_position'][noise],
        df_data['visual_noise_position'][noise]
    ])

# Combine sensor noise conditions (vector and visual)
sensor_noise_data = {}
for noise in labels:
    sensor_noise_data[noise] = np.vstack([
        df_data['vector_noise_distributed'][noise],
        df_data['visual_noise_distributed'][noise]
    ])


# 1. Effect of noise level on learning speed (slope)


def compute_slope(runs_data):
    """Compute slope (block 5 - block 1) for each run"""
    return runs_data[:, 1] - runs_data[:, 0]

def compute_auc(runs_data):
    """Compute area under the curve (sum across all blocks) for each run"""
    return runs_data.sum(axis=1)

# Test across all conditions
all_slopes_by_noise = {}
all_auc_by_noise = {}
for noise in labels:
    # Combine all conditions for this noise level
    all_runs = np.vstack([
        df_data['vector_noise_position'][noise],
        df_data['visual_noise_position'][noise],
        df_data['vector_noise_distributed'][noise],
        df_data['visual_noise_distributed'][noise]
    ])
    all_slopes_by_noise[noise] = compute_slope(all_runs)
    all_auc_by_noise[noise] = compute_auc(all_runs)

# One-way ANOVA across noise levels - SLOPE
slope_groups = [all_slopes_by_noise[n] for n in labels]
f_stat, p_val = f_oneway(*slope_groups)
print("One-way ANOVA across noise levels (SLOPE):")
print(f"  F-statistic = {f_stat:.4f}, p-value = {p_val:.6f}")

if p_val < 0.05:
    print("  *** Significant effect of noise level on learning speed (p < 0.05)")
else:
    print("No significant effect of noise level on learning speed")

# Show means
print("\nMean slopes by noise level:")
for noise in labels:
    mean_slope = np.mean(all_slopes_by_noise[noise])
    print(f"  Noise {noise}: {mean_slope:.4f}")

# One-way ANOVA across noise levels - AUC
print("\nOne-way ANOVA across noise levels (AUC - learning efficiency):")
auc_groups = [all_auc_by_noise[n] for n in labels]
f_stat_auc, p_val_auc = f_oneway(*auc_groups)
print(" F-statistic = {f_stat_auc:.4f}, p-value = {p_val_auc:.6f}")

if p_val_auc < 0.05:
    print("*** Significant effect of noise level on learning efficiency (p < 0.05)")
else:
    print("No significant effect of noise level on learning efficiency")

# Show means
print("\nMean AUC by noise level:")
for noise in labels:
    mean_auc = np.mean(all_auc_by_noise[noise])
    print(f"  Noise {noise}: {mean_auc:.4f}")


# 2. Effect of noise type (pose vs. sensor) on learning speed

print("\n\n2. EFFECT OF NOISE TYPE (POSE vs. SENSOR) ON LEARNING SPEED")
print("-" * 70)

# Combine across all noise levels for each type (excluding 0.0)
labels_no_zero = [n for n in labels if n != '0.0']
pose_slopes = np.concatenate([compute_slope(pose_noise_data[n]) for n in labels_no_zero])
sensor_slopes = np.concatenate([compute_slope(sensor_noise_data[n]) for n in labels_no_zero])
pose_auc = np.concatenate([compute_auc(pose_noise_data[n]) for n in labels_no_zero])
sensor_auc = np.concatenate([compute_auc(sensor_noise_data[n]) for n in labels_no_zero])

# Independent t-test (overall) - SLOPE
t_stat, p_val = ttest_ind(pose_slopes, sensor_slopes)
print("Independent t-test - SLOPE (Pose vs. Sensor noise, excluding noise level 0.0):")
print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.6f}")
print(f"  Pose noise mean slope: {np.mean(pose_slopes):.4f}")
print(f"  Sensor noise mean slope: {np.mean(sensor_slopes):.4f}")

if p_val < 0.05:
    print("  *** Significant difference in learning speed (p < 0.05)")
else:
    print("  No significant difference in learning speed")

# Independent t-test (overall) - AUC
t_stat_auc, p_val_auc = ttest_ind(pose_auc, sensor_auc)
print("\nIndependent t-test - AUC (Pose vs. Sensor noise, excluding noise level 0.0):")
print(f"  t-statistic = {t_stat_auc:.4f}, p-value = {p_val_auc:.6f}")
print(f"  Pose noise mean AUC: {np.mean(pose_auc):.4f}")
print(f"  Sensor noise mean AUC: {np.mean(sensor_auc):.4f}")

if p_val_auc < 0.05:
    print("  *** Significant difference in learning efficiency (p < 0.05)")
else:
    print("  No significant difference in learning efficiency")

# Test separately for vector and visual
print("\n2a. Vector conditions only (pose vs. sensor):")
vector_pose_slopes = np.concatenate([compute_slope(df_data['vector_noise_position'][n]) for n in labels_no_zero])
vector_sensor_slopes = np.concatenate([compute_slope(df_data['vector_noise_distributed'][n]) for n in labels_no_zero])
vector_pose_auc = np.concatenate([compute_auc(df_data['vector_noise_position'][n]) for n in labels_no_zero])
vector_sensor_auc = np.concatenate([compute_auc(df_data['vector_noise_distributed'][n]) for n in labels_no_zero])

t_stat, p_val = ttest_ind(vector_pose_slopes, vector_sensor_slopes)
print(f"  SLOPE - t-statistic = {t_stat:.4f}, p-value = {p_val:.6f}")
print(f"    Vector pose mean slope: {np.mean(vector_pose_slopes):.4f}")
print(f"    Vector sensor mean slope: {np.mean(vector_sensor_slopes):.4f}")
if p_val < 0.05:
    print("    *** Significant difference (p < 0.05)")
else:
    print("    No significant difference")

t_stat_auc, p_val_auc = ttest_ind(vector_pose_auc, vector_sensor_auc)
print(f"  AUC - t-statistic = {t_stat_auc:.4f}, p-value = {p_val_auc:.6f}")
print(f"    Vector pose mean AUC: {np.mean(vector_pose_auc):.4f}")
print(f"    Vector sensor mean AUC: {np.mean(vector_sensor_auc):.4f}")
if p_val_auc < 0.05:
    print("    *** Significant difference (p < 0.05)")
else:
    print("    No significant difference")

print("\n2b. Visual conditions only (pose vs. sensor):")
visual_pose_slopes = np.concatenate([compute_slope(df_data['visual_noise_position'][n]) for n in labels_no_zero])
visual_sensor_slopes = np.concatenate([compute_slope(df_data['visual_noise_distributed'][n]) for n in labels_no_zero])
visual_pose_auc = np.concatenate([compute_auc(df_data['visual_noise_position'][n]) for n in labels_no_zero])
visual_sensor_auc = np.concatenate([compute_auc(df_data['visual_noise_distributed'][n]) for n in labels_no_zero])

t_stat, p_val = ttest_ind(visual_pose_slopes, visual_sensor_slopes)
print(f"  SLOPE - t-statistic = {t_stat:.4f}, p-value = {p_val:.6f}")
print(f"    Visual pose mean slope: {np.mean(visual_pose_slopes):.4f}")
print(f"    Visual sensor mean slope: {np.mean(visual_sensor_slopes):.4f}")
if p_val < 0.05:
    print("    *** Significant difference (p < 0.05)")
else:
    print("    No significant difference")

t_stat_auc, p_val_auc = ttest_ind(visual_pose_auc, visual_sensor_auc)
print(f"  AUC - t-statistic = {t_stat_auc:.4f}, p-value = {p_val_auc:.6f}")
print(f"    Visual pose mean AUC: {np.mean(visual_pose_auc):.4f}")
print(f"    Visual sensor mean AUC: {np.mean(visual_sensor_auc):.4f}")
if p_val_auc < 0.05:
    print("    *** Significant difference (p < 0.05)")
else:
    print("    No significant difference")


# 3. Effect of noise type on final block performance

print("\n\n3. EFFECT OF NOISE TYPE ON FINAL BLOCK PERFORMANCE")
print("-" * 70)

# Get final block (block 5) performance
pose_final = np.concatenate([pose_noise_data[n][:, -1] for n in labels])
sensor_final = np.concatenate([sensor_noise_data[n][:, -1] for n in labels])

# Independent t-test
t_stat, p_val = ttest_ind(pose_final, sensor_final)
print("Independent t-test (Pose vs. Sensor noise):")
print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.6f}")
print(f"  Pose noise mean final performance: {np.mean(pose_final):.4f}")
print(f"  Sensor noise mean final performance: {np.mean(sensor_final):.4f}")

if p_val < 0.05:
    print("  *** Significant difference in final performance (p < 0.05)")
else:
    print("  No significant difference in final performance")


# 4. Effect of input type (vector vs. visual) within each noise type

print("\n\n4. EFFECT OF INPUT TYPE (VECTOR vs. VISUAL) WITHIN NOISE TYPE")
print("-" * 70)

#POSE NOISE: Vector vs Visual
print("\n4a. POSE NOISE: Vector vs. Visual (excluding noise level 0.0)")
print("-" * 70)

# Slopes and AUC
vector_pose_slopes = np.concatenate([compute_slope(df_data['vector_noise_position'][n]) for n in labels_no_zero])
visual_pose_slopes = np.concatenate([compute_slope(df_data['visual_noise_position'][n]) for n in labels_no_zero])
vector_pose_auc = np.concatenate([compute_auc(df_data['vector_noise_position'][n]) for n in labels_no_zero])
visual_pose_auc = np.concatenate([compute_auc(df_data['visual_noise_position'][n]) for n in labels_no_zero])

# Final block performance
vector_pose_final = np.concatenate([df_data['vector_noise_position'][n][:, -1] for n in labels_no_zero])
visual_pose_final = np.concatenate([df_data['visual_noise_position'][n][:, -1] for n in labels_no_zero])

# Test SLOPE
t_stat, p_val = ttest_ind(vector_pose_slopes, visual_pose_slopes)
print(f"SLOPE - t-statistic = {t_stat:.4f}, p-value = {p_val:.6f}")
print(f"  Vector mean slope: {np.mean(vector_pose_slopes):.4f}")
print(f"  Visual mean slope: {np.mean(visual_pose_slopes):.4f}")
if p_val < 0.05:
    print("  *** Significant difference (p < 0.05)")
else:
    print("  No significant difference")

# Test AUC
t_stat_auc, p_val_auc = ttest_ind(vector_pose_auc, visual_pose_auc)
print(f"\nAUC - t-statistic = {t_stat_auc:.4f}, p-value = {p_val_auc:.6f}")
print(f"  Vector mean AUC: {np.mean(vector_pose_auc):.4f}")
print(f"  Visual mean AUC: {np.mean(visual_pose_auc):.4f}")
if p_val_auc < 0.05:
    print("  *** Significant difference (p < 0.05)")
else:
    print("  No significant difference")

# Test final block performance
t_stat_final, p_val_final = ttest_ind(vector_pose_final, visual_pose_final)
print(f"\nFINAL BLOCK PERFORMANCE - t-statistic = {t_stat_final:.4f}, p-value = {p_val_final:.6f}")
print(f"  Vector mean final performance: {np.mean(vector_pose_final):.4f}")
print(f"  Visual mean final performance: {np.mean(visual_pose_final):.4f}")
if p_val_final < 0.05:
    print("  *** Significant difference (p < 0.05)")
else:
    print("  No significant difference")


print("\n\n4b. SENSOR NOISE: Vector vs. Visual (excluding noise level 0.0)")
print("-" * 70)

# Slopes and AUC
vector_sensor_slopes = np.concatenate([compute_slope(df_data['vector_noise_distributed'][n]) for n in labels_no_zero])
visual_sensor_slopes = np.concatenate([compute_slope(df_data['visual_noise_distributed'][n]) for n in labels_no_zero])
vector_sensor_auc = np.concatenate([compute_auc(df_data['vector_noise_distributed'][n]) for n in labels_no_zero])
visual_sensor_auc = np.concatenate([compute_auc(df_data['visual_noise_distributed'][n]) for n in labels_no_zero])

# Final block performance
vector_sensor_final = np.concatenate([df_data['vector_noise_distributed'][n][:, -1] for n in labels_no_zero])
visual_sensor_final = np.concatenate([df_data['visual_noise_distributed'][n][:, -1] for n in labels_no_zero])

# Test SLOPE
t_stat, p_val = ttest_ind(vector_sensor_slopes, visual_sensor_slopes)
print(f"SLOPE - t-statistic = {t_stat:.4f}, p-value = {p_val:.6f}")
print(f"  Vector mean slope: {np.mean(vector_sensor_slopes):.4f}")
print(f"  Visual mean slope: {np.mean(visual_sensor_slopes):.4f}")
if p_val < 0.05:
    print("  *** Significant difference (p < 0.05)")
else:
    print("  No significant difference")

# Test AUC
t_stat_auc, p_val_auc = ttest_ind(vector_sensor_auc, visual_sensor_auc)
print(f"\nAUC - t-statistic = {t_stat_auc:.4f}, p-value = {p_val_auc:.6f}")
print(f"  Vector mean AUC: {np.mean(vector_sensor_auc):.4f}")
print(f"  Visual mean AUC: {np.mean(visual_sensor_auc):.4f}")
if p_val_auc < 0.05:
    print("  *** Significant difference (p < 0.05)")
else:
    print("  No significant difference")

# Test final block performance
t_stat_final, p_val_final = ttest_ind(vector_sensor_final, visual_sensor_final)
print(f"\nFINAL BLOCK PERFORMANCE - t-statistic = {t_stat_final:.4f}, p-value = {p_val_final:.6f}")
print(f"  Vector mean final performance: {np.mean(vector_sensor_final):.4f}")
print(f"  Visual mean final performance: {np.mean(visual_sensor_final):.4f}")
if p_val_final < 0.05:
    print("  *** Significant difference (p < 0.05)")
else:
    print("  No significant difference")

