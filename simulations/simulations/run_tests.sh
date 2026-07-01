#!/bin/bash


for sim_type in intermittent continuous
do
    for add_noise_to in vector
    do
        for noise_type in position
        do
           for noise in 0.0
           do
               for input_type in vector_only visual_only both
               do
                   for i in {1..10}
                   do 
                       python test_agents.py $i $noise $add_noise_to $noise_type $input_type $sim_type
                   done
               done
           done
       done
    done
done
