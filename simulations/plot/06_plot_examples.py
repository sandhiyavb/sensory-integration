#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import scipy.ndimage as nd
import matplotlib.pyplot as plt

def clean_fields(fields) : 
    fields = np.squeeze(fields)
    n_units = fields.shape[2]
    cleaned_fields = np.zeros(fields.shape)
    for f,i in zip(np.rollaxis(fields,2),range(n_units)) :
        for hd,j in zip(f,range(6)) :
            hd = hd.reshape(25,25)
            hd = nd.gaussian_filter(hd,sigma=1)
            hd = np.where(hd<0.50*np.max(hd),0,
                                    hd)
            hd = hd.flatten()
            cleaned_fields[j,:,i] = hd
    return cleaned_fields

noises = np.arange(0.0,0.6,0.1)


inp = 'both'
simtype = 'certain_task'
prefs = [f'../data/{simtype}/noise_simulations/vector_noise_position/',
         f'../data/{simtype}/noise_simulations/visual_noise_position/',
         f'../data/{simtype}/noise_simulations/vector_noise_distributed/',
         f'../data/{simtype}/noise_simulations/visual_noise_distributed/']



noise = noises[0]
pref = prefs[0]
noise_folder = os.path.join(pref, f"noise_{noise:.1f}")

file_path = os.path.join(noise_folder, f"classified_fields_{inp}.csv")
df = pd.read_csv(file_path)

run = 1
unit_type = "Place"

df_run = df[(df['run']==run) & (df['unit_type']==unit_type)]
df_elements = df_run
unit_ids = df_elements['unit'].to_list()

activations_pref = pref + f'noise_{noise}/run_{run}/activations/'
activations_files = [activations_pref + 'activations_both.npy',
                     activations_pref + 'activations_vector_only.npy',
                     activations_pref + 'activations_visual_only.npy']

files=[np.squeeze(np.load(f)) for f in activations_files]

unit_activations = [np.take(file, unit_ids, axis=2) for file in files]
cleaned_fields = [np.mean(clean_fields(fields),axis=0) for fields in unit_activations]

for i in range(len(unit_ids)) :
    fig, ax = plt.subplots(1,3, dpi=300)
    example = ax[0].imshow(nd.filters.gaussian_filter(cleaned_fields[0][:,i].reshape(25,25), sigma=1), cmap='turbo')
    ax[1].imshow(nd.filters.gaussian_filter(cleaned_fields[1][:,i].reshape(25,25), sigma=1), cmap='turbo')
    ax[2].imshow(nd.filters.gaussian_filter(cleaned_fields[2][:,i].reshape(25,25), sigma=1), cmap='turbo')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')

cbar = fig.colorbar(example, orientation='horizontal')
fig.savefig('figs/colorbar.svg', format='svg', dpi=300)