# basic imports
import blender_frontend
import numpy as np
import math
import ast
# blender imports
import bge
import mathutils

def add_gaussian_noise(value, mean, var, scale_by_max=None) :    
    sigma = var**0.5
    if scale_by_max is not None and scale_by_max!=0.0 : 
        sigma = sigma * abs(scale_by_max)
    
    if sigma != 0 : 
        gauss = np.random.normal(mean, sigma)
    else : 
        gauss = 0.0
    return value + gauss

class FrontendNoise(blender_frontend.BlenderFrontend) : 
    
    def __init__(self, control_buffer_size=1000):
        
        super().__init__(control_buffer_size=control_buffer_size)
        self.functions['teleportXY'] = self.teleportXY
        self.functions['show_object'] = self.show_object
        self.functions['teleportXYyaw'] = self.teleportXYyaw
        
    def teleportXY(self, data) :
        objectName, xStr, yStr = data
        x=float(xStr)
        y=float(yStr)
        object_name=self.scene.objects[objectName]
        object_name.worldPosition.x = x
        object_name.worldPosition.y = y
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
        
    def teleportXYyaw(self, data) :
        objectName, xStr, yStr, yawStr = data
        x=float(xStr)
        y=float(yStr)
        yaw=float(yawStr)
        object_name=self.scene.objects[objectName]
        object_name.worldPosition.x = x
        object_name.worldPosition.y = y
        euler = mathutils.Euler((0.0, 0.0, yaw / 180 * math.pi), 'XYZ')
        object_name.worldOrientation = euler.to_matrix()
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
        
    def show_object(self, data) :
        objectName, state = data
        state = (state=='True')
        object=self.scene.objects[objectName]
        object.setVisible(state)
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
        
    def stepSimNoPhysics(self, data):
        '''
        This function teleports the robot and propels the simulation by one time step.
        
        | **Args**
        | velocityLeftStr:              The desired left wheel velocity.
        | velocityRightStr:             The desired right wheel velocity.
        '''
        # split data string
        newXStr, newYStr, newYawStr, noise = data
        # recover position and orientation
        newX, newY, newYaw = float(newXStr), float(newYStr), float(newYawStr)
        camera_noise=float(noise)
        # update simulation time
        self.simulationTime += self.dT
        # retrieve the robot's current position
        currentX, currentY, currentZ = self.robotSupport.worldPosition[0], self.robotSupport.worldPosition[1], self.robotSupport.worldPosition[2]
        # switch off physics for object in teleport
        self.robotSupport.setLinearVelocity([0.0, 0.0, 0.0], False)
        self.robotSupport.setAngularVelocity([0.0, 0.0, 0.0], False)
        # tie wheels to the robot's support
        self.leftWheel.setParent(self.robotSupport)
        self.rightWheel.setParent(self.robotSupport)
        # update the robot's position
        self.robotSupport.worldPosition.x = newX
        self.robotSupport.worldPosition.y = newY
        # update the robot's orientation
        euler = mathutils.Euler((0.0, 0.0, newYaw / 180 * math.pi), 'XYZ')
        self.robotSupport.worldOrientation = euler.to_matrix()
        # untie the wheels from the robot's support
        self.leftWheel.removeParent()
        self.rightWheel.removeParent()
        
        #calculate noisy inputs x,y,yaw
        noisy_x = add_gaussian_noise(newX, 0.0, camera_noise,
                                     scale_by_max=None)
        noisy_y = add_gaussian_noise(newY, 0.0, camera_noise,
                                     scale_by_max=None)
        noisy_yaw = add_gaussian_noise(newYaw, 0.0, camera_noise,
                                       scale_by_max=60.0)
        noisy_yaw = noisy_yaw%360.0
        camera = self.scene.objects['Camera']
        camera.worldPosition.x = noisy_x
        camera.worldPosition.y = noisy_y
        euler = mathutils.Euler((0.0, 0.0, noisy_yaw / 180 * math.pi), 'XYZ')
        camera.worldOrientation = euler.to_matrix()
        
        # update BGE clock
        bge.logic.setClockTime(self.simulationTime)
        # retrieve headings
        headingX = self.robotSupport.worldOrientation[0][0]
        headingY = self.robotSupport.worldOrientation[1][0]
        # refresh canvases
        self.refresh_canvases()
        # send control data
        sendString = '%.5f:%.3f,%.3f,%.3f,%.3f:%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (self.simulationTime, self.robotSupport.worldPosition[0], self.robotSupport.worldPosition[1], headingX, headingY, self.sensorArray[0], self.sensorArray[1], self.sensorArray[2], self.sensorArray[3], self.sensorArray[4], self.sensorArray[5], self.sensorArray[6], self.sensorArray[7])
        self.controller['controlConnection'].send(sendString.encode('utf-8'))
        # send video data
        self.controller['videoConnection'].send(self.bufFront)
        self.controller['videoConnection'].send(self.bufLeft)
        self.controller['videoConnection'].send(self.bufRight)
        self.controller['videoConnection'].send(self.bufBack)

BF = FrontendNoise()
BF.main_loop()