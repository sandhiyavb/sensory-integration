#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
from itertools import combinations
from statsmodels.stats.multitest import multipletests

file_name = '/training_log.csv'
folders = ['../data/intermittent/noise_simulations/visual_noise_distributed/noise_0.0',
           '../data/continuous/noise_simulations/visual_noise_distributed/noise_0.0']

colors = ['#F18F01','#048BA8']
labels = ['intermittent', 'continuous']
    
fig, ax = plt.subplots(figsize=(3.5,2.5),dpi=300)

def p_to_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

def add_sig_bar(ax, x1, x2, y, p, h=0.03):
    """Draw significance bar between two bars"""
    
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c='black')
    ax.text((x1+x2)/2, y+h, p_to_star(p), ha='center', va='bottom')

def auc_first_n(training_data, n_trials=300):
    """Compute AUC per run for first n trials"""
    runs = np.unique(training_data[:,0])
    
    aucs = []
    for r in runs:
        rewards = training_data[np.where(training_data[:,0]==r)][:,3][:n_trials]
        auc = np.trapz(rewards)  # numerical integration
        aucs.append(auc)
    
    return np.array(aucs)

def smooth_reward(reward, integrator=20) : 
    """ Returns the episode reward suitable for plotting by integrating over 
        the last n values as specified on the plot
        
    """
    y = []
    for i in range(len(reward) - integrator):
        temp = np.sum(reward[i:i+integrator])/integrator
        y.append(temp)
        
    x = np.arange(len(y))
    return np.array([x,y])

def mean_std_reward(training_data, cutoff=3999) : 
    """ Returns mean and standard deviation of reward for multiple training
    runs """
    runs = np.unique(training_data[:,0])
    all_rewards = np.stack([training_data[np.where(training_data[:,0]==i)][:,3][:cutoff] for i in runs])
    mean_reward = np.mean(all_rewards, axis=0)
    std_dev = np.std(all_rewards, axis=0)
    
    return np.array([mean_reward,std_dev])

def mean_sem_reward(training_data, cutoff=3999): 
    """Returns mean and SEM of reward for multiple training runs"""
    
    runs = np.unique(training_data[:,0])
    
    all_rewards = np.stack([
        training_data[np.where(training_data[:,0]==i)][:,3][:cutoff]
        for i in runs
    ])
    
    mean_reward = np.mean(all_rewards, axis=0)
    n = all_rewards.shape[0]
    sem = np.std(all_rewards, axis=0, ddof=1) / np.sqrt(n)
    
    return np.array([mean_reward, sem])

def plot_reward(mean_reward,axis,color,label,std_reward=None) : 
    """ Plots learning curve with standard deviation on given matplotlib 
    axis object"""
    
    axis.plot(mean_reward[0],mean_reward[1],c=color, alpha = 0.8,label=label)
    if std_reward is not None : 
        axis.fill_between(std_reward[0], 
                         mean_reward[1] + std_reward[1], 
                         mean_reward[1] - std_reward[1], alpha = 0.2, 
                         facecolor = color)
    
    axis.set_xlabel('Trials')
    axis.set_ylabel('Average trial reward')

def success_rate(test_data_csv) : 
    test_data = pd.read_csv(test_data_csv,sep='\t')
    counts = []
    groups = test_data.groupby('run')
    for g in groups : 
        try : 
            counts.append((100-g[1].steps.value_counts()[100])/100)
        except :
            counts.append(1.0)
        
    return np.mean(counts), np.std(counts)

def success_rate_per_run(test_data_csv):
    test_data = pd.read_csv(test_data_csv, sep='\t')
    counts = []
    
    groups = test_data.groupby('run')
    for g in groups:
        try:
            counts.append((100 - g[1].steps.value_counts()[100]) / 100)
        except:
            counts.append(1.0)
    
    return np.array(counts)

def plot_test_bars(data_sources, spacing=0, label=None, ax=None, color='grey',
                   input_types = ['Visual only', 'Vector only', 'Both']) : 
    means = []
    stds   = []
    
    for d in data_sources : 
        rates = success_rate_per_run(d)
        mean = np.mean(rates)
        n = len(rates)
        sem = np.std(rates) / np.sqrt(n)

        means.append(mean)
        stds.append(sem)
    
    # Create lists for the plot
    #stds = [np.zeros(len(stds)), stds]
    x_pos = np.arange(len(input_types))
    means = np.array(means)
    stds = np.array(stds)
    lower_err = np.minimum(stds, means)
    upper_err = np.minimum(stds, 1 - means)
    
    yerr = np.vstack([lower_err, upper_err])
    # Build the plot    
    ax.bar(x_pos+spacing, means, yerr=yerr, align='center',width=0.2, alpha=0.9, ecolor='#333333', linewidth=0.5,
           capsize=1.5, color=color,label=label, edgecolor = "gray")
    
    ax.set_ylabel('Proportion successful test trials')
    ax.set_xlabel('Noise')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(input_types)
    #ax.set_title('Test performance')
    ax.set_ylim(top=1.5)
    #ax.yaxis.grid(True)
    
    return means, stds

auc_results = []

for i in range(len(folders)) : 
    training_data = np.genfromtxt(folders[i]+file_name)[1:]
    aucs = auc_first_n(training_data, 300)
    auc_results.append(aucs)
    
    data = mean_sem_reward(training_data, 1000)
    mean_reward = smooth_reward(data[0])
    std_reward = smooth_reward(data[1])
    plot_reward(mean_reward, ax, colors[i], labels[i],std_reward)
    

t_stat, p_val = ttest_ind(auc_results[0], auc_results[1], equal_var=False)

print("\nAUC comparison (first 300 trials):")
print(f"Intermittent mean: {np.mean(auc_results[0]):.3f}")
print(f"Continuous mean:   {np.mean(auc_results[1]):.3f}")
print(f"p-value: {p_val:.5f}")

plt.legend()
plt.rcParams.update({'font.size': 10}) 



#Comparison of test performance
test_logs = ["test_log_visual_only.csv", "test_log_vector_only.csv",
             "test_log_both.csv"]

test_data = [[f+'/'+test for test in test_logs] for f in folders]
test_results = []

for i in range(len(folders)):  #intermittent/continuous
    condition_results = []
    
    for d in test_data[i]:  #visual/vector/both
        rates = success_rate_per_run(d)
        condition_results.append(rates)
    
    test_results.append(condition_results)

conditions = ['Visual only', 'Vector only', 'Both']

print("\nIntermittent vs Continuous (per condition):")

p_vals_inter_vs_cont = []
labels_inter_vs_cont = []

for i, cond in enumerate(['Visual only', 'Vector only', 'Both']):
    t_stat, p_val = ttest_ind(test_results[0][i],
                             test_results[1][i],
                             equal_var=False)
    
    p_vals_inter_vs_cont.append(p_val)
    labels_inter_vs_cont.append(cond)
    


print("\nBetween conditions (within each training type):")

p_vals_between = []
labels_between = []

for t_idx, label in enumerate(labels):  # intermittent / continuous
    for (i, j) in combinations(range(3), 2):
        t_stat, p_val = ttest_ind(test_results[t_idx][i],
                                 test_results[t_idx][j],
                                 equal_var=False)
        
        p_vals_between.append(p_val)
        labels_between.append(f"{label}: {i} vs {j}")


# Intermittent vs Continuous
reject_ic, pvals_ic_corr, _, _ = multipletests(
    p_vals_inter_vs_cont,
    method='fdr_bh'
)

# Between conditions
reject_between, pvals_between_corr, _, _ = multipletests(
    p_vals_between,
    method='bonferroni'
)

print("\nIntermittent vs Continuous (corrected):")

for cond, p_raw, p_corr, sig in zip(labels_inter_vs_cont,
                                    p_vals_inter_vs_cont,
                                    pvals_ic_corr,
                                    reject_ic):
    
    print(f"{cond}: raw p = {p_raw:.5f}, corrected p = {p_corr:.5f}, significant = {sig}")
    
print("\nBetween conditions (corrected):")

for label, p_raw, p_corr, sig in zip(labels_between,
                                     p_vals_between,
                                     pvals_between_corr,
                                     reject_between):
    
    print(f"{label}: raw p = {p_raw:.5f}, corrected p = {p_corr:.5f}, significant = {sig}")
    
    
fig2, ax2 = plt.subplots(figsize=(3.5,2.5), dpi=300)

spacing = [-0.1,0.1]
means, stds = [], []
for i in range(len(folders)) : 
    mean, std = plot_test_bars(test_data[i], spacing[i], labels[i], ax2, colors[i])
    means.append(mean)
    stds.append(std)
    plt.legend(loc='lower center', ncols=2)
    plt.rcParams.update({'font.size': 10}) 

# x positions
x_pos = np.arange(3)  # 3 conditions
x_inter = x_pos + spacing[0]
x_cont  = x_pos + spacing[1]

conditions = ['Visual only', 'Vector only', 'Both']

idx = 0  # index for corrected p-values

for i, cond in enumerate(['Visual only', 'Vector only', 'Both']):
    
    # positions of the two bars
    x1 = x_pos[i] + spacing[0]  # intermittent
    x2 = x_pos[i] + spacing[1]  # continuous
    
    # height = max of the two bars + margin
    y = max(means[0][i], means[1][i]) + 0.05
    
    # corrected p-value
    p = pvals_ic_corr[i]
    
    add_sig_bar(ax2, x1, x2, y, p)
    
for t_idx, label in enumerate(labels):  # intermittent / continuous
    
    # choose correct x positions
    x_current = x_inter if t_idx == 0 else x_cont
    
    # base height (top of bars)
    base_heights = means[t_idx]
    
    # stack bars upward
    height_offset = 0.08
    
    for k, (i, j) in enumerate(combinations(range(3), 2)):
        
        x1 = x_current[i]
        x2 = x_current[j]
        
        y = max(base_heights[i], base_heights[j]) + height_offset * (k+1)
        
        p = pvals_between_corr[idx]
        
        if p < 0.05:
            add_sig_bar(ax2, x1, x2, y, p, h=0.05)
        
        idx += 1
    
#save figures    
if not os.path.exists('figs') :
    os.makedirs('figs')
fig.savefig('figs/learning_curve.svg', format='svg', dpi=300)
fig2.savefig('figs/test_performance_hc.svg', format='svg', dpi=300)