#!/bin/bash

for sim_type in intermittent continuous
do
    for add_noise_to in vector visual
    do
        for noise_type in position distributed
        do
           for noise in 0.1 0.2 0.3 0.4 0.5
           do
               for i in {1..10}
               do 
                   python train_agents.py $i $noise $add_noise_to $noise_type $sim_type 2>/dev/null
               done
           done
        done
    done
done
