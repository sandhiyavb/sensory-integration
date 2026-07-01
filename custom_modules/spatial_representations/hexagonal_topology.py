import numpy as np
from cobel.spatial_representations.topology_graphs.simple_topology_graph  import HexagonalGraph
import gym
import random
import PyQt5 as qt
import pyqtgraph as qg
import pyqtgraph.functions
import csv

class HexagonalGraphAllocentric(HexagonalGraph) :
    
    def generate_behavior_from_action(self, action) :
        
        callback_value = dict()
        trans_actions = np.arange(0,6)
        rot_actions   = np.arange(6,12)
        orientations = np.arange(0,360,60)
        hd_dict = dict(zip(rot_actions, orientations))
        
        if action!='reset' : 
            if action in trans_actions : 
                node_id = self.nodes[self.current_node].neighbors[action].index
                if node_id != -1 : 
                    self.move_to_node(node_id, self.head_direction, self.noise)
            
            if action in rot_actions : 
                node_id = self.current_node
                self.head_direction = hd_dict[action]
                if node_id != -1 : 
                    self.move_to_node(node_id, self.head_direction, self.noise)
                
            else  :
                self.next_node = self.current_node
                self.world_module.goalReached = True
        
        else :                 
            node_id = random.choice(self.start_nodes)
            random_direction = random.choice(orientations)
            self.head_direction = random_direction
            self.move_to_node(node_id, random_direction, self.noise)
        
        self.current_node = self.next_node
        callback_value['currentNode'] = self.nodes[self.current_node]
        
        return callback_value
    
    def move_to_node(self, node_id, angle=0.0, noise=0.0) :
        
        self.next_node = node_id
        next_node_pos = np.array([self.nodes[self.next_node].x, 
                          self.nodes[self.next_node].y])
    
        self.world_module.actuateRobot(np.array([next_node_pos[0],
                                         next_node_pos[1],
                                         angle, noise]))
    
        self.current_node = self.next_node
        self.observation_module.update()
        
        if self.visual_output : 
            self.update_position_marker([next_node_pos[0], next_node_pos[1],
                                         np.cos(np.deg2rad(angle)),  
                                         np.sin(np.deg2rad(angle))])
            
            if hasattr(qt.QtGui, 'QApplication'):
                if qt.QtGui.QApplication.instance() is not None:
                    qt.QtGui.QApplication.instance().processEvents()
            else:
                if qt.QtWidgets.QApplication.instance() is not None:
                    qt.QtWidgets.QApplication.instance().processEvents()
    
    def get_action_space(self) :
        
        return gym.spaces.Discrete(12)
    
    def save_node_xy(self, filename):
        
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idx",
                             "x","y"])
            
            for node in self.nodes:
                writer.writerow([node.index,
                                 node.x,
                                 node.y])
                
        print(f"Saved nodes to: {filename}")
            