# basic imports
import os
import numpy as np
import pyqtgraph as qg
import pandas as pd
import sys
# tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
# framework imports

from custom_modules.renderers.renderers import BlenderOnlineRenderer
from custom_modules.observations.image_observations import ImageObservationFOV
from custom_modules.observations.vector_observations import VectorObservation
from cobel.observations.dictionary_observations import DictionaryObservations
from custom_modules.spatial_representations.hexagonal_topology import HexagonalGraphAllocentric
from cobel.agents.multi_dict_dqn import DQNAgentMultiModal
from cobel.interfaces.oai_gym_interface import OAIGymInterface
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor

from aux.callbacks import RewardFunction, TrainingLogger

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = True

network = "../../networks/multimodal_vector_visual_large_allo.json"
json_file   = open(network, 'r')
loaded_model_json = json_file.read()
json_file.close()  
model = model_from_json(loaded_model_json)    
print(model.summary())

df = pd.DataFrame(columns=['run','trial','steps','reward']) 
pref = 'data/rochefort/'
   
def single_run(run=1, noise=0.1):
    '''
    This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    '''
    
    data_folder = pref+'noise_'+noise+'/'
    run_data_folder = data_folder+'run_'+run
    print("Run : ", run)    
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Demo: DQN')
        
    if not os.path.exists(run_data_folder) :
        os.makedirs(run_data_folder)
    if not os.path.exists(run_data_folder+'/weights') :
        os.makedirs(run_data_folder+'/weights')
    if not os.path.exists(run_data_folder+'/activations') :     
       os.makedirs(run_data_folder+'/activations')
    
    # determine demo scene path
    demo_scene = os.path.abspath(__file__).split('emergent_spatial_representations')[0] + '/emergent_spatial_representations/worlds/guidance.blend'
    print(demo_scene)
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = BlenderOnlineRenderer(demo_scene)
    vector_observation = VectorObservation(None, main_window, noise=(0.0, float(noise)),
                                           vector_encoding='allocentric')
    image_observation = ImageObservationFOV(modules['world'], main_window, visual_output,
                                                 imageDims=(72, 12), view_angle=240.0, noise=(0.0,0.1))
    modules['observation'] = DictionaryObservations({'image_input':image_observation,
                                                     'vector_input':vector_observation})   
    modules['spatial_representation'] = HexagonalGraphAllocentric(n_nodes_x=5, n_nodes_y=5,
                                                       n_neighbors=6, goal_nodes=[11],
                                                       visual_output=True, 
                                                       world_module=modules['world'],
                                                       use_world_limits=True, 
                                                       observation_module=modules['observation'], 
                                                       rotation=True)
    vector_observation.add_topology_graph(modules['spatial_representation'])
    modules['spatial_representation'].set_visual_debugging(main_window)
    reward_function = RewardFunction(step_reward=-1.0, goal_reward=1.0)
    modules['rl_interface'] = OAIGymInterface(modules, visual_output, 
                                              reward_function.reward_callback)
    
    # amount of trials
    number_of_trials = 500
    # maximum steps per trial
    max_steps = 100
    record_weights_at = np.arange(0,550,50)
    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials, main_window, visual_output, 
                                   [-max_steps, 10])
    
    data_logger = TrainingLogger(run_data_folder, record_weights_at=record_weights_at)
    # initialize RL agent
    rl_agent = DQNAgentMultiModal(modules['rl_interface'], 3000, 0.3, model=model, 
                                custom_callbacks={'on_trial_end': [reward_monitor.update,
                                                                   data_logger.record_training_data,
                                                                   data_logger.save_intermediate_weights]})
                                                                   
    #if we'd like to record intermediate activations, define a keras function
    analysis_model = rl_agent.model
    activations_layer = 'analysis'
    keras_function = K.function([analysis_model.get_layer('image_input').input, 
                                          analysis_model.get_layer('vector_input').input],
                                         [analysis_model.get_layer(activations_layer).output])
    data_logger.add_keras_function(keras_function)
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent
    
    # let the agent learn, with extremely large number of allowed maximum steps
    rl_agent.train(number_of_trials, max_steps)
    
    x = np.array(data_logger.training_data)
    x = np.hstack((int(run)*np.ones(number_of_trials).reshape(number_of_trials,1), x))
    logs = df.append(pd.DataFrame(x, columns=df.columns), ignore_index=True)
    logs.to_csv(data_folder+'training_log.csv', sep='\t', encoding='utf-8',index=False,
              mode='a', header=not os.path.exists(data_folder+'training_log.csv'))
    
    rl_agent.agent.save_weights(run_data_folder+'/weights/weights_trial_{}.h5'.format(number_of_trials))
    # clear keras session (for performance)

    K.clear_session()
    
    # stop simulation
    modules['world'].stopBlender()
    
    # and also stop visualization
    if visual_output:
        main_window.close()

if __name__ == '__main__':
            
    single_run(sys.argv[1], sys.argv[2])
# clear keras session (for performance)
    K.clear_session()
