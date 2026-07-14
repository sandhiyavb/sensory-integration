#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from analysis.learning_and_behavior import direct_paths_to_goal
from scipy.stats import wilcoxon

def start_valid_island(a, accuracy, window_size=3):
    m = a>=accuracy
    me = np.r_[False,m,False]
    idx = np.flatnonzero(me[:-1]!=me[1:])
    lens = idx[1::2]-idx[::2]
    return idx[::2][(lens >= window_size).argmax()]

def add_significance_bar(ax, x1, x2, y, p_value, h=3):
    """Draw significance bar with stars"""
    
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c='black')
    
    #Convert p-value to stars
    if p_value < 0.001:
        text = '***'
    elif p_value < 0.01:
        text = '**'
    elif p_value < 0.05:
        text = '*'
    else:
        text = 'n.s'
    
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom')

#Comparison of test performance
folders = ['../data/stackman/sham_first/',
           '../data/stackman/vestibular_first/']

training_data = "training_log.csv"

means_df, std_df = [], []

first_criterion = []
for f in folders : 
    data = np.genfromtxt(f+training_data)[1:]
    
    runs = np.unique(data[:,0])
    steps = np.stack([data[np.where(data[:,0]==i)][:,2] for i in runs])
     
    n_trials = []
    for single_trial in steps : 
        direct_path = 15
        window = 16
        accuracy = 1
        
        accuracies = []
        
        for i in range(len(single_trial)-window) : 
            sample = single_trial[i:i+window] 
            sample_accuracy = (sample <= direct_path).sum()/window
            accuracies.append(sample_accuracy)
            
        accuracies = np.array(accuracies)
        trials_to_criterion = start_valid_island(accuracies, accuracy,3) + window
        n_trials.append(trials_to_criterion)
    first_criterion.append(n_trials)

#Comparison of test performance
folders = ['../data/stackman/sham_second/',
           '../data/stackman/vestibular_second/']


training_data = "training_log.csv"

means_df, std_df = [], []

second_criterion = []
for f in folders : 
    data = np.genfromtxt(f+training_data)[1:]
    
    runs = np.unique(data[:,0])
    steps = np.stack([data[np.where(data[:,0]==i)][:,2] for i in runs])
     
    n_trials = []
    for single_trial in steps : 
        direct_path = 10
        window = 16
        accuracy = 1
        
        accuracies = []
        
        for i in range(len(single_trial)-window) : 
            sample = single_trial[i:i+window] 
            sample_accuracy = (sample <= direct_path).sum()/window
            accuracies.append(sample_accuracy)
            
        accuracies = np.array(accuracies)
        trials_to_criterion = start_valid_island(accuracies, accuracy,3) + window
        n_trials.append(trials_to_criterion)
    second_criterion.append(n_trials)

#STATSTESTS
first_sham = np.array(first_criterion[0])
first_ves  = np.array(first_criterion[1])

second_sham = np.array(second_criterion[0])
second_ves  = np.array(second_criterion[1])

#Wilcoxon paired tests
stat1, p_first = wilcoxon(first_sham, first_ves)
stat2, p_second = wilcoxon(second_sham, second_ves)

print("\nTrials to criterion stats:")
print(f"First condition:  p = {p_first:.5f}")
print(f"Second condition: p = {p_second:.5f}")

means = {'sham' : [np.mean(first_criterion[0]), np.mean(second_criterion[0])],
         'vestibular' : [np.mean(first_criterion[1]), np.mean(second_criterion[1])]}

stds = {
    'sham': [
        np.std(first_criterion[0], ddof=1) / np.sqrt(len(first_criterion[0])),
        np.std(second_criterion[0], ddof=1) / np.sqrt(len(second_criterion[0]))
    ],
    'vestibular': [
        np.std(first_criterion[1], ddof=1) / np.sqrt(len(first_criterion[1])),
        np.std(second_criterion[1], ddof=1) / np.sqrt(len(second_criterion[1]))
    ]
}

criterion = ["First", "Second"]

x = np.arange(len(criterion)) + 1  # the label locations
width = 0.25  # the width of the bars
offset = 0.2
multiplier = 0
legend_offset = -0.3

fig, ax = plt.subplots(figsize=(3,2.5), dpi=300)
color = {'sham' :'black', 'vestibular': 'white'}
for attribute, measurement in means.items():
    
    rects = ax.bar(x + offset + multiplier * width, measurement, width-0.01, 
                   label=attribute, color=color[attribute], linewidth=1, edgecolor='black')
    upper_error = stds[attribute]
    std = [np.zeros(len(upper_error)), upper_error]
    ax.errorbar(x + offset + multiplier * width,
                measurement,
                yerr=upper_error,
                fmt='none',
                ecolor='black',
                capsize=5,
                elinewidth=1)
    multiplier += 1
    
# positions must match your bars
x_positions = x + offset

# First condition comparison
y_max1 = max(means['sham'][0], means['vestibular'][0]) + 10
add_significance_bar(ax,
                     x_positions[0],
                     x_positions[0] + width,
                     y_max1,
                     p_first)

# Second condition comparison
y_max2 = max(means['sham'][1], means['vestibular'][1]) + 10
add_significance_bar(ax,
                     x_positions[1],
                     x_positions[1] + width,
                     y_max2,
                     p_second, h=6)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Trials to criterion')
ax.set_xticks(x + width + offset, criterion)
ax.legend(loc='upper right', bbox_to_anchor=(1 - legend_offset, 1))
ax.set_ylim(0,120)

plt.show()
plt.tight_layout()
plt.rcParams.update({'font.size': 10})
fig.savefig('figs/stackman.svg', format='svg', dpi=300)