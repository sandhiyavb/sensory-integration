#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

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
    
