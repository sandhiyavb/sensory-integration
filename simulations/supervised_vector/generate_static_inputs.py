
''' Generate input images on a grid for
    recording activations
'''
from aux.helpers import resize_images, limit_field_of_view
import numpy as np
from scipy.stats import truncnorm
from custom_modules.renderers.renderers import BlenderOnlineRenderer
import os

scene = os.path.abspath(__file__).split('emergent_spatial_representations')[0] + '/emergent_spatial_representations/worlds/guidance.blend'
input_size = (72, 12)
fov        = 240.0

def calculate_angle_ref(x1, y1, x2, y2, ref=(1,0)) : 

    #translate to have tail at (0,0)
    current_vector = (x2 - x1, y2 - y1)
    angle = np.arctan2(current_vector[1], current_vector[0]) - np.arctan2(ref[1],ref[0])
    if angle >= 0.0 : 
        angle = np.rad2deg(angle)
    else : 
        angle = 360.0 + np.rad2deg(angle)
    return angle

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

def add_gaussian_noise_to_value(value, mean, var, scale_by_value=True) :    
    sigma = var**0.5
    if scale_by_value : 
        sigma = sigma * value
    gauss = np.random.normal(mean,sigma)
            
    return value + gauss

def gen_spatial_images(world, angles, grid_size ,radial=False,
                              fov=fov, input_size = input_size,
                              noise=None) :
    """
    Specify a grid size and environment to generate 
    input images at grid points with (x,y) position.

    Note : saves original 256x64 images from Blender.
    Remember to resize them to match your network 
    input before using.
    """
    
    world_limits = world.getLimits()
    x_limits = [world_limits[0,0], world_limits[0,1]]
    y_limits = [world_limits[1,0], world_limits[1,1]]
    print(x_limits, y_limits)
    X = np.linspace(x_limits[0], x_limits[1], grid_size[0])
    Y = np.linspace(y_limits[0], y_limits[1], grid_size[1])

    xx,yy = np.meshgrid(X,Y)
    grid_points = np.vstack((xx.flatten(), yy.flatten())).T
    
    if radial : 
        rad   = np.linspace(0, x_limits[1], num=grid_size[0])
        theta = np.linspace(0, 2*np.pi, num=grid_size[1])    
        rr,tt = np.meshgrid(rad,theta)
        xx,yy = rr*np.cos(tt),rr*np.sin(tt)
        grid_points = np.vstack((xx.flatten(),yy.flatten())).T
    
    images = []
        
    for angle in angles :   
        angle_images = []  
        for point in grid_points :
            observation = world.step_simulation_without_physics(point[0],point[1],angle)
            if noise is not None : 
                observation = add_gaussian_noise(observation[3], noise[0], noise[1])   
            else : observation = observation[3]
            image = observation.astype('float32')/255.0          
            angle_images.append(image)    
        resized_images = resize_images(angle_images, input_size[0], input_size[1])
        cropped_images = limit_field_of_view(resized_images, input_size[0], fov)        
        images.append(cropped_images)
    return np.array(images)


def gen_spatial_allocentric_vectors(world, angles, grid_size, radial=False,
                                    noise=None) : 
    world_limits = world.getLimits()
    x_limits = [world_limits[0,0], world_limits[0,1]]
    y_limits = [world_limits[1,0], world_limits[1,1]]
    X = np.linspace(x_limits[0], x_limits[1], grid_size[0])
    Y = np.linspace(y_limits[0], y_limits[1], grid_size[1])

    xx,yy = np.meshgrid(X,Y)
    grid_points = np.vstack((xx.flatten(), yy.flatten())).T
    max_distance = np.sqrt((x_limits[0] - x_limits[1])**2 + (y_limits[0] - y_limits[1])**2)
    print(max_distance)
    
    if radial : 
        rad   = np.linspace(0, x_limits[1], num=grid_size[0])
        theta = np.linspace(0, 2*np.pi, num=grid_size[1])    
        rr,tt = np.meshgrid(rad,theta)
        xx,yy = rr*np.cos(tt),rr*np.sin(tt)
        grid_points = np.vstack((xx.flatten(),yy.flatten())).T
        
    vectors = []        
    for angle in angles :   
        angle_vectors = []  
        for point in grid_points :   
            #goal is at the center of the arena! 
            allocentric_angle = calculate_angle_ref(point[0], point[1],0.0,0.0)
            noisy_angle = angle
            if noise is not None : 
                allocentric_angle = add_gaussian_noise_to_value(allocentric_angle, noise[0], 
                                                      noise[1])
                noisy_angle = add_gaussian_noise_to_value(angle, noise[0], 
                                                      noise[1])
            allocentric_angle = allocentric_angle / 360.0
            
            distance   = np.sqrt((point[0])**2 + (point[1])**2)
            if noise is not None : 
                distance = add_gaussian_noise_to_value(distance, noise[0], noise[1])
            distance = distance / max_distance #normalize distance
            
            
            angle_vectors.append([distance, allocentric_angle, noisy_angle/360.0])
        vectors.append(angle_vectors)
    return np.array(vectors)

angles = np.linspace(0,300,6,dtype='int64')
resolution = (15,15)

world = BlenderOnlineRenderer(scene)
image_data = gen_spatial_images(world, angles, resolution, noise=None)
vector_data = gen_spatial_allocentric_vectors(world, angles, resolution,
                                              noise=None)
np.save('inputs/input_images_test.npy', image_data)
np.save('inputs/output_vectors_test.npy', vector_data)
