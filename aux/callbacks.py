import numpy as np

class RewardFunction():
    
    def __init__(self, step_reward=0.0, goal_reward=10.0, shock_nodes=[],
                 shock_value=-20.0):
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.shock_value = shock_value
        self.shock_nodes = shock_nodes        
    
    def update_step_reward(self, step_reward):
        self.step_reward = step_reward
        
    def update_goal_reward(self, goal_reward): 
        self.goal_reward = goal_reward
        
    def reward_callback(self, values): 
        reward = self.step_reward
        end_trial = False
        
        if values['currentNode'] in self.shock_nodes :
            reward = self.shock_value
            end_trial = False
            
        if values['currentNode'].goalNode:
            reward = self.goal_reward
            end_trial = True
    
        return reward, end_trial
        
class TrainingLogger():
    
    def __init__(self, data_folder, record_weights_at=None, 
                 record_activations_at=None, keras_function=None, 
                 input_files=None, compute_zero_inputs=False):
        self.training_data = []
        self.data_folder = data_folder
        self.record_activations_at = record_activations_at
        self.record_weights_at = record_weights_at
        self.keras_function = keras_function
        self.input_files = input_files
        self.all_activations = []
        self.compute_zero_inputs = compute_zero_inputs
        if self.compute_zero_inputs :
            self.zero_inputs = [np.zeros(inp.shape) for inp in self.input_files]
            
    def record_training_data(self, logs):
        trial = logs['trial']
        self.training_data.append([trial, logs['steps'], logs['trial_reward']])

                
    def save_intermediate_weights(self, logs) : 
        trial    = logs['trial']
        rl_agent = logs['rl_parent']
        if trial in self.record_weights_at : 
            rl_agent.agent.save_weights(self.data_folder+"/weights/weights_trial_{}.h5".format(trial))
    
    def save_intermediate_activations(self, logs) :
        #TODO : implement exception if keras_function is None or if input_files is None
        #TODO : check that all inputs have the same no. of heading directions 
        trial = logs['trial']
        headings = len(self.input_files[0])
        if trial in self.record_activations_at :
            activations = []
            zero_activations = [[] for inp in self.input_files] 
            for i in range(headings) : 
                inputs = [np.expand_dims(inp[i],1) for inp in self.input_files]
                activations.append(self.keras_function(inputs))
                if self.compute_zero_inputs :
                    zero_inps = [np.expand_dims(inp[i],1) for inp in self.zero_inputs]
                    #TODO : make general
                    zero_activations[0].append(self.keras_function([zero_inps[0],
                                                                    inputs[1]]))
                    zero_activations[1].append(self.keras_function([inputs[0],
                                                zero_inps[1]]))
            
            self.all_activations.append(activations)
            np.save(self.data_folder+"/activations/activations_trial_{}.npy".format(trial),
                    activations)
            if self.compute_zero_inputs : 
                np.save(self.data_folder+"/activations/activations_input_0_trial_{}.npy".format(trial),
                        zero_activations[0])
                np.save(self.data_folder+"/activations/activations_input_1_trial_{}.npy".format(trial),
                        zero_activations[1])
                
            
    def add_keras_function(self, keras_function) : 
        self.keras_function = keras_function
        