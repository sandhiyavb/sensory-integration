#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib_venn import venn3, venn3_circles

#Plot what the place fields in the both condition change to with only vector/visual inputs
simtypes = ['uncertain_task', 'certain_task']
inputs = ['both','vector_only','visual_only']
noise = 0.0

all_results = []  
plt.rcParams.update({'font.size': 25})
for simtype in simtypes:
    pref = f'../data/{simtype}/noise_simulations/visual_noise_distributed/' #with zero noise doesn't matter which one
    noise_folder = os.path.join(pref, f"noise_{noise:.1f}")


    dfs = []
    
    for inp in inputs:
        file_path = os.path.join(noise_folder, f"classified_fields_{inp}.csv")
        dfs.append(pd.read_csv(file_path))
        
    place_both = dfs[0][dfs[0]['unit_type']=='Place'].drop(['trial','unit_type'], axis=1)
    place_vector_only = dfs[1][dfs[1]['unit_type']=='Place'].drop(['trial','unit_type'], axis=1)
    place_visual_only = dfs[2][dfs[2]['unit_type']=='Place'].drop(['trial','unit_type'], axis=1)
    

    set1 = set(place_both.apply(lambda row: (row['run'], row['unit']), axis=1))
    set2 = set(place_vector_only.apply(lambda row: (row['run'], row['unit']), axis=1))
    set3 = set(place_visual_only.apply(lambda row: (row['run'], row['unit']), axis=1))
    
    #Create the Venn diagram
    plt.figure(figsize=(8, 6), dpi=300)
    v = venn3([set1, set2, set3], ('Both inputs', 'Vector only', 'Visual only'),
              set_colors=('lightgrey', 'grey', 'dimgray'))

    circles = venn3_circles([set1, set2, set3], linestyle='solid', linewidth=2, color='black')
    
    for text in v.set_labels:
        text.set_color('black')
    
    plt.show()