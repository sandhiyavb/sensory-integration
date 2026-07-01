from cobel.observations.image_observations import ImageObservationBaseline
# basic imports
import numpy as np
from PyQt5.QtCore import QRectF
# open-cv
import cv2
from scipy.stats import truncnorm
from tensorflow.keras.models import load_model
import gym

def add_gaussian_noise(observation, mean, var,
                       scale_by_mean=True, use_trunc_norm=True) :    
    row, col, ch= observation.shape
    sigma = var**0.5
    #add scaled noise
    if scale_by_mean : 
        sigma = sigma * np.mean(observation)

    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    
    if use_trunc_norm : 
        lower, upper = 0, 255
        gauss_trunc = truncnorm((lower - mean) / sigma, (upper - mean) / sigma,
                                loc=mean, scale=sigma)
        gauss = gauss_trunc.rvs((row,col,ch))
    
    return observation + gauss
    
def crop_image(image, image_dim, view_angle) : 
    keep_pixels = image_dim[0] * view_angle / 360.0
    remove_pixels = image_dim[0] - keep_pixels
    rm = np.ceil(remove_pixels/2).astype('int32')
    cropped_image = image[:,rm:rm+np.ceil(keep_pixels).astype('int32')]
    return cropped_image
    
def rotate_panorama(image, degrees):
    height, width = image.shape[:2]
    shift = int((degrees / 360.0) * width)
    if shift > 0:  # Rotate to the right
        rotated_image = np.hstack((image[:, -shift:], image[:, :-shift]))
    else:  # Rotate to the left
        shift = abs(shift)
        rotated_image = np.hstack((image[:, shift:], image[:, :shift]))    
    return rotated_image

class ImageObservationFOV(ImageObservationBaseline) : 
    
    def __init__(self, world, guiParent, visualOutput=True, imageDims=(30, 1),
                 view_angle=360.0, noise=None, rotate_by=None) :
        super().__init__(world, guiParent, visualOutput=visualOutput, imageDims=imageDims)
        self.view_angle = view_angle 
        self.observation = crop_image(image=np.zeros((self.imageDims[1],
                                                      self.imageDims[0],3)),
                                         image_dim=imageDims, 
                                         view_angle=self.view_angle)
        self.noise = noise
        self.rotate_by = rotate_by
        # _observe determines if the observation is actually recorded from the
        #environment (true observation) or replaced by dummy data.
        #This should normally always be set to True, but can be used to temporarily
        #turn off the observation to simulate the loss of sensory signal
        self._observe = True
        
    def update(self):
        '''
        This function processes the raw image data and updates the current observation.
        '''
        # the observation is plainly the robot's camera image data
        if self._observe : observation = self.worldModule.envData['imageData']
        else : observation = np.zeros((self.imageDims[1], self.imageDims[0],3))
        # display the observation camera image
        if self.visualOutput:
            imageData = observation
            self.cameraImage.setOpts(axisOrder='row-major')
            imageData = imageData[:,:,::-1]
            self.cameraImage.setImage(imageData)
            imageScale = 1.0
            self.cameraImage.setRect(QRectF(0.0, 0.0, imageScale, imageData.shape[0]/imageData.shape[1]*imageScale))
        # scale the one-line image to further reduce computational demands
        observation = cv2.resize(observation, dsize=self.imageDims)
        #resize according to field of view
                
        if self._observe : 
            if self.noise is not None : 
                observation = add_gaussian_noise(observation, self.noise[0], self.noise[1], 
                                                 use_trunc_norm=True)
            if self.rotate_by is not None : 
                observation = rotate_panorama(observation, self.rotate_by)
            
        observation = crop_image(observation, self.imageDims, self.view_angle)
        observation.astype('float32') 
        observation = observation/255.0
        #observation = (observation - np.min(observation))/(np.max(observation) - np.min(observation))
        # display the observation camera image reduced to one line
        if self.visualOutput:
            imageData = observation
            self.observationImage.setOpts(axisOrder='row-major')
            imageData = imageData[:,:,::-1]
            self.observationImage.setImage(imageData)
            imageScale = 1.0
            self.observationImage.setRect(QRectF(0.0, -0.1, imageScale, 
                                                 imageData.shape[0]/imageData.shape[1]*imageScale))
        self.observation = observation
        
    def set_observation_state(self, state) :
        self._observe = state
        
        
class ImageFromMemory(ImageObservationFOV):
    
    def __init__(self, world, guiParent, visualOutput=True, imageDims=(30, 1),
                 view_angle=360.0, noise=(0.0,0.0)) :
        
        
        super().__init__(world, guiParent, visualOutput=visualOutput, imageDims=imageDims,
                         view_angle=view_angle,noise=noise)
        self.memory = np.zeros((self.imageDims[1], self.imageDims[0],3))
    
    def update(self) : 
        if self._observe : observation = self.memory
        else : observation = np.zeros((self.imageDims[1], self.imageDims[0],3))
        # display the observation camera image
        if self.visualOutput:
            imageData = observation
            self.cameraImage.setOpts(axisOrder='row-major')
            imageData = imageData[:,:,::-1]
            self.cameraImage.setImage(imageData)
            imageScale = 1.0
            self.cameraImage.setRect(QRectF(0.0, -0.5, imageScale, imageData.shape[0]/imageData.shape[1]*imageScale))
        # scale the one-line image to further reduce computational demands
        observation = cv2.resize(observation, dsize=self.imageDims)
        #resize according to field of view
        observation = crop_image(observation, self.imageDims, self.view_angle)
        observation.astype('float32')  
        observation = observation/255.0
        if self._observe : 
            observation = add_gaussian_noise(observation, self.noise[0], self.noise[1],
                                             use_trunc_norm=False)
        # display the observation camera image reduced to one line
        if self.visualOutput:
            imageData = observation
            self.observationImage.setOpts(axisOrder='row-major')
            imageData = imageData[:,:,::-1]
            self.observationImage.setImage(imageData)
            imageScale = 1.0
            self.observationImage.setRect(QRectF(0.0, -0.6, imageScale, 
                                                 imageData.shape[0]/imageData.shape[1]*imageScale))
        self.observation = observation
        
    def update_memory(self) : 
        self.memory = self.worldModule.envData['imageData']
    

class ImagetoVectorEstimate(ImageObservationBaseline) : 

    def __init__(self, world, guiParent, visualOutput=False, imageDims=(72,12), view_angle=360.0, 
             prediction_model=None, vector_encoding='allocentric') :
        """
        insert docstring
        """
        
        super().__init__(world, guiParent, visualOutput=visualOutput, imageDims=imageDims)
        self.prediction_model=load_model(prediction_model)
        self._observe = True
        self.vector_encoding = vector_encoding
        self.view_angle = view_angle 

        if self.vector_encoding == 'egocentric' :
            self.observation     = np.array([0.0,0.0])
        elif self.vector_encoding == 'allocentric' :
            self.observation     = np.array([0.0,0.0,0.0]) 
            
    def update(self) : 
        
        if self._observe : 
            observation = self.worldModule.envData['imageData']
            observation = cv2.resize(observation, dsize=self.imageDims)
            observation = crop_image(observation, self.imageDims, self.view_angle) 
            observation.astype('float32')
            observation = observation/255.0
            
            vector_estimate = self.prediction_model.predict(np.expand_dims(observation,axis=0))
            self.observation = np.squeeze(vector_estimate)
        else : 
            if self.vector_encoding == 'egocentric' :
                self.observation     = np.array([0.0,0.0])
            elif self.vector_encoding == 'allocentric' :
                self.observation     = np.array([0.0,0.0,0.0]) 
            
    def getObservationSpace(self) : 
        
        '''
        This function returns the observation space for the given observation class.
        '''
        if self.vector_encoding=='egocentric' :
            return gym.spaces.Box (low=0.0, high=1.0, shape=(2,))
        else : 
            return gym.spaces.Box (low=0.0, high=1.0, shape=(3,))
    
    def set_observation_state(self, state) :
        self._observe = state
