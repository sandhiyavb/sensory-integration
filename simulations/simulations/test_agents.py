#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:53:30 2026

@author: sandhiya
"""

# basic imports
import os
import numpy as np
import pyqtgraph as qg
import pandas as pd
import sys
import json
# tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json

#logging time
import logging
import time

# framework imports

from cobel.observations.dictionary_observations import DictionaryObservations
from custom_modules.renderers.renderers import BlenderOnlineRenderer
from custom_modules.observations.image_observations import ImageObservationFOV
from custom_modules.observations.vector_observations import VectorObservation

from custom_modules.spatial_representations.hexagonal_topology import HexagonalGraphAllocentric
from cobel.agents.multi_dict_dqn import DQNAgentMultiModal
from cobel.interfaces.oai_gym_interface import OAIGymInterface
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor

from aux.callbacks import RewardFunction, TrainingLogger


visual_output = True

df = pd.DataFrame(columns=['run','trial','steps','reward']) 

def update_distributed_noise(json_filename, new_rate, output_filename=None):
    with open(json_filename, 'r') as f:
        model_json = json.load(f)

    modified = False
    #Search for GaussianDropout layer
    for layer in model_json['config']['layers']:
        if layer['class_name'] == 'GaussianDropout':
            old_rate = layer['config']['rate']
            if old_rate != new_rate:
                print(f"Updating GaussianDropout rate: {old_rate} to {new_rate}")
                layer['config']['rate'] = new_rate
                modified = True
            else:
                print(f"Rate already {old_rate}, file not updated")

    if not modified:
        print("No changes made")
        return model_from_json(json.dumps(model_json))

    output_filename = output_filename or json_filename
    with open(output_filename, 'w') as f:
        json.dump(model_json, f, indent=2)
    print(f"Model saved to {output_filename}")

    #Rebuild and return updated model
    return model_from_json(json.dumps(model_json))

def single_run(run=1, noise=0.1, add_noise_to='vector', noise_type='distributed', 
               input_type='both', sim_type='continuous'):
    '''
    This method performs a single experimental run, i.e. one experiment. 
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    '''
    
    logging.basicConfig(
    filename="runtime_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s")

    start = time.time()
    
    mode = add_noise_to + ' ' + noise_type
    data_folder = f"data/{sim_type}/{add_noise_to}_noise_{noise_type}/noise_{noise}/" 
    
    input_types = ['visual_only', 'vector_only', 'both', 'random']
    if input_type not in input_types : 
        raise ValueError("Input type must be one of the following : {}".format(input_types))    
    
    network_dict = {'visual distributed' : '../../networks/multimodal_vector_visual_large_allo.json',
                    'visual position' : '../../networks/multimodal_vector_visual_large_allo.json',
                    'vector distributed' : '../../networks/multimodal_vector_visual_large_allo_noise.json',
                    'vector position' : '../../networks/multimodal_vector_visual_large_allo.json'}

    
    network = network_dict[mode]
    
    if mode == 'vector distributed' : 
        vector_distributed_noise = 0.0
        model = update_distributed_noise(network, vector_distributed_noise)
        
    else : 
        vector_distributed_noise = None
        json_file   = open(network, 'r')
        loaded_model_json = json_file.read()
        json_file.close()  
        model = model_from_json(loaded_model_json)   
        
    print(model.summary())
    
    run_data_folder = data_folder+'run_'+run
    print("Run : ", run)    
    if not os.path.exists(run_data_folder) :
        os.makedirs(run_data_folder)
    if not os.path.exists(run_data_folder+'/weights') :
        os.makedirs(run_data_folder+'/weights')
    if not os.path.exists(run_data_folder+'/activations') :     
       os.makedirs(run_data_folder+'/activations')
    
    np.random.seed()
    main_window = None
    if visual_output:
        main_window = qg.GraphicsWindow(title='Demo: DQN')
        
    
    # determine demo scene path
    demo_scene = os.path.abspath(__file__).split('emergent_spatial_representations')[0] + '/emergent_spatial_representations/worlds/guidance-noise.blend'
    print(demo_scene)
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = BlenderOnlineRenderer(demo_scene)
    
        
    vector_observation = VectorObservation(None, main_window, noise=None,
                                           vector_encoding='allocentric')
    

    image_observation = ImageObservationFOV(modules['world'], main_window, visual_output,
                                                 imageDims=(72, 12), view_angle=240.0, noise=None)
    
    
    
    
    modules['observation'] = DictionaryObservations({'image_input':image_observation,
                                                     'vector_input':vector_observation})
    

    modules['spatial_representation'] = HexagonalGraphAllocentric(n_nodes_x=5, n_nodes_y=5,
                                                       n_neighbors=6, goal_nodes=[11],
                                                       visual_output=True, 
                                                       world_module=modules['world'],
                                                       use_world_limits=True, 
                                                       observation_module=modules['observation'], 
                                                       rotation=True, noise=0.0)
    
    
    vector_observation.add_topology_graph(modules['spatial_representation'])
    modules['spatial_representation'].set_visual_debugging(main_window)
    reward_function = RewardFunction(step_reward=-1.0, goal_reward=1.0)
    modules['rl_interface'] = OAIGymInterface(modules, visual_output, 
                                              reward_function.reward_callback)
    
    
    
    if input_type in ["both","random"] : 
        print("RUNNING TEST WITH VISUAL AND VECTOR INPUTS")
        
    if input_type == "visual_only" : 
        print("RUNNING TEST WITH VISUAL INPUTS ONLY")
        vector_observation.set_observation_state(False)
        
    if input_type == "vector_only" : 
        print("RUNNING TEST WITH VECTOR INPUTS ONLY")
        image_observation.set_observation_state(False)
        
    # amount of trials
    number_of_trials = 100
    # maximum steps per trial
    max_steps = 100
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, 
                                   [-max_steps, 10])
    
    data_logger = TrainingLogger(run_data_folder)
    # initialize RL agent
    rl_agent = DQNAgentMultiModal(modules['rl_interface'], 3000, 0.3, model=model, 
                                custom_callbacks={'on_trial_end': [reward_monitor.update,
                                                                   data_logger.record_training_data]})
                                                                   

    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent
    
    weights = run_data_folder+'/weights/weights_trial_4000.h5'
    # let the agent learn, with extremely large number of allowed maximum steps
    rl_agent.test(number_of_trials, max_steps, weights=weights)
    
    x = np.array(data_logger.training_data)
    x = np.hstack((int(run)*np.ones(number_of_trials).reshape(number_of_trials,1), x))
    logs = df.append(pd.DataFrame(x, columns=df.columns), ignore_index=True)
    logs.to_csv(data_folder+'test_log_{}.csv'.format(input_type), 
                sep='\t', encoding='utf-8',index=False,
              mode='a', 
              header=not os.path.exists(data_folder+'test_log_{}.csv'.format(input_type)))
    
    # clear keras session (for performance)

    K.clear_session()
    
    # stop simulation
    modules['world'].stopBlender()
    
    # and also stop visualization
    if visual_output:
        main_window.close()
        
    elapsed = time.time() - start
    logging.info(f"{mode} -- {noise} -- {run} finished in {elapsed:.2f} seconds")

if __name__ == '__main__':
            
    single_run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
# clear keras session (for performance)
    K.clear_session()
