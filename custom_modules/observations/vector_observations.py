import numpy as np
import gym


def calculate_angle_ref(x1, y1, x2, y2, ref=(1,0)) : 

    #translate to have tail at (0,0)
    current_vector = (x2 - x1, y2 - y1)
    angle = np.arctan2(current_vector[1], current_vector[0]) - np.arctan2(ref[1],ref[0])
    if angle >= 0.0 : 
        angle = np.rad2deg(angle)
    else : 
        angle = 360.0 + np.rad2deg(angle)
    return angle

def add_gaussian_noise(value, mean, var, scale_by_value=True) :    
    sigma = var**0.5
    if scale_by_value : 
        if value != 0.0 :
            sigma = sigma * value
    gauss = np.random.normal(mean,sigma)
        
    return value + gauss

class VectorObservation() : 
    
    def __init__(self, topology, gui_parent, vector_encoding='egocentric',
                 noise=None, rotate_by=None) :
        
        encoding_list = ['egocentric', 'allocentric']
        if vector_encoding not in encoding_list : 
            raise ValueError("Vector encoding must be either 'egocentric' or 'allocentric'")
        
        self.topology        = topology
        self.gui_parent      = gui_parent
        self.vector_encoding = vector_encoding
        self.noise = noise
        self.rotate_by = rotate_by
        
        if self.vector_encoding == 'egocentric' :
            self.observation     = np.array([0.0,0.0])
        elif self.vector_encoding == 'allocentric' :
            self.observation     = np.array([0.0,0.0,0.0]) 
            
        # _observe determines if the observation is actually recorded from the
        #environment (true observation) or replaced by dummy data.
        #This should normally always be set to True, but can be used to temporarily
        #turn off the observation to simulate the loss of sensory signal
        self._observe = True
        
    def add_topology_graph(self, topology_graph) : 
        self.topology = topology_graph
    
    def update(self) : 
        '''
        Updates the observation. For observing a vector to the goal, this will
        compute the vector between the current node on the topology graph and
        the goal node.

        Returns
        -------
        None.

        '''
        current_node  = self.topology.current_node
        goal_nodes    = self.topology.goal_nodes
        
        current_x = self.topology.nodes[current_node].x
        current_y = self.topology.nodes[current_node].y
        
        #For now, we'll assume a single goal node is present. 
        #This code will need to be modified to accomodate multiple goal
        #locations
        
        goal_x = self.topology.nodes[goal_nodes[0]].x
        goal_y = self.topology.nodes[goal_nodes[0]].y
        
        distance   = np.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)
        
        #this is the angle of the line connecting the goal and current location
        #w.r.t a reference vector pointing due east
        allocentric_angle = calculate_angle_ref(current_x, current_y, goal_x, goal_y)

        #this is the current heading direction of the agent
        head_direction    = self.topology.head_direction
        #egocentric direction - we calculate the direction of the goal wrt the
        #heading direction
        
        hd_rad = np.deg2rad(head_direction)
        
        egocentric_angle = calculate_angle_ref(current_x,current_y, goal_x, goal_y,
                                           ref=(np.cos(hd_rad), np.sin(hd_rad)))
        if self.noise is not None : 
            egocentric_angle = add_gaussian_noise(egocentric_angle, self.noise[0], 
                                                  self.noise[1])
        egocentric_angle = egocentric_angle / 360.0
        max_distance = np.sqrt((self.topology.nodes[0].x - self.topology.nodes[-1].x)**2 + (self.topology.nodes[0].y - self.topology.nodes[-1].y)**2)
        
        s = np.array([distance/max_distance, allocentric_angle/360.0, head_direction/360.0])

        if self.noise is not None : 
            distance = add_gaussian_noise(distance, self.noise[0], 
                                                  self.noise[1])
        distance = distance / max_distance #normalize distance
        
        if self.vector_encoding=='egocentric' : 
            if self._observe : 
                observation = np.array([distance, egocentric_angle])
                #observation = add_gaussian_noise(observation, self.noise[0], self.noise[1])
            else : observation = np.array([0.0,0.0])            
            self.observation = observation
            
        if self.vector_encoding=='allocentric' : 
            if self._observe : 
                if self.noise is not None : 
                    head_direction = add_gaussian_noise(head_direction, self.noise[0], 
                                                          self.noise[1])

                    allocentric_angle = add_gaussian_noise(allocentric_angle, self.noise[0], 
                                                          self.noise[1])
                    
                if self.rotate_by is not None : 
                        
                    #head_direction = head_direction + self.rotate_by
                    allocentric_angle = allocentric_angle + self.rotate_by
                    if allocentric_angle < 0.0 :
                        allocentric_angle = 360.0 + allocentric_angle
                    
                    allocentric_angle = allocentric_angle % 360.0
                
                observation = np.array([distance, allocentric_angle / 360.0,
                                             head_direction / 360.0])
                n = s - observation
                snr = np.linalg.norm(s)**2 / np.linalg.norm(n)**2
                #print(f"s:{s}, n:{n}, SNR:{snr}")
            else :                 
                if self.noise is not None : 
                    observation = np.array([add_gaussian_noise(0.0, self.noise[0], self.noise[1]),
                                            add_gaussian_noise(0.0, self.noise[0], self.noise[1]),
                                            add_gaussian_noise(0.0, self.noise[0], self.noise[1])])
                else : observation = np.array([0.0,0.0,0.0])
            self.observation = observation


    def set_observation_state(self, state) :
        self._observe = state
    
    def getObservationSpace(self):
        
        '''
        This function returns the observation space for the given observation class.
        '''
        if self.vector_encoding=='egocentric' :
            return gym.spaces.Box (low=0.0, high=1.0, shape=(2,))
        else : 
            return gym.spaces.Box (low=0.0, high=1.0, shape=(3,))
    
    
    
