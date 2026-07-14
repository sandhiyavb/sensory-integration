#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

prefs = ['../data/conflict/rotations/vector_rotation/']

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


df_vector = []
df_visual = []
df_both = []

for i in range(len(prefs)) : 
    pref = prefs[i]
    folders = [f for f in os.listdir(pref)]
    folders.sort(key=int)
    
    angles = np.arange(-180,210,30)
    test_logs = ['test_log_vector_only_4000.csv','test_log_visual_only_4000.csv', 
                 'test_log_both_4000.csv']
    
    test_data = [[pref+f+'/'+test for test in test_logs] for f in folders]
    
    for t in test_data : 
        mean_vector, std_vector = success_rate(t[0])
        df_vector.append(mean_vector)
        mean_visual, std_visual = success_rate(t[1])
        df_visual.append(mean_visual)
        mean_both, std_both = success_rate(t[2])
        df_both.append(mean_both)
        
    labels = angles
    

    fig, ax = plt.subplots(figsize=(9,6),dpi=300)
    
    ax.plot(angles, df_vector, c='#454545', label='vector only',
            linewidth=2.5, marker='o', markersize=5)

    ax.plot(angles, df_both, color='#BFAF80', label='both',
            linewidth=2.5, marker='^', markersize=5)
    ax.hlines(y=np.mean(df_visual),xmin=-200, xmax=200, color='#E3B23C',linestyle='--',
              label='visual only')

    ax.set_xlabel('Mismatch in Degrees', fontsize=15)
    ax.set_ylabel('Proportion of Successful Test Trials', fontsize=15)
    

    ax.spines[['right', 'top']].set_visible(False)

    plt.rcParams.update({'font.size': 15})
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4),
               shadow=True, ncols=3, frameon=False)
    

    plt.tight_layout()

    fig.savefig('figs/test_performance_rotation.svg', format='svg', dpi=300)