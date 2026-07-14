import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

folder = '../data/conflict/activations_vector'
runs = np.arange(1,16,1)
rots = np.arange(-180,210,30)

file = 'classified_fields_both_inputs.csv'
#get all place fields for a specific run
df = pd.read_csv(os.path.join(folder,file))
df_place = df[df['unit_type']=='Place'].drop('unit_type',axis=1)

#get common units from all runs
results = []

for run, df_run in df_place.groupby('run'):
    rotations = df_run['rotation'].unique()

    common_units = None
    for rotation in rotations:
        units = set(df_run[df_run['rotation'] == rotation]['unit'])
        if common_units is None:
            common_units = units
        else:
            common_units &= units  # intersection with next rotation

    results.append({
        'run': run,
        'common_units': sorted(list(common_units)),
        'count': len(common_units)
    })

df_common = pd.DataFrame(results)
run = 14
units = df_common[df_common['run']==run]['common_units'].to_list()[0]
activations = [os.path.join(folder, f'run_{run}','activations',f'activations_{rot}.npy') for rot in rots]

for unit in units : 
    activations_unit = [np.squeeze(np.load(activation))[:,:,unit] for activation in activations]
    images = [np.reshape(np.mean(field, axis=0), (25,25)) for field in activations_unit]
    images = [gaussian_filter(img, sigma=2) for img in images]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_axis_off() 
    
    #Plot the center image (rotation = 0)
    center_idx = np.where(rots == 0)[0][0]
    center_img = OffsetImage(images[center_idx], zoom=1.9, cmap='turbo')
    ab = AnnotationBbox(center_img, (0, 0), frameon=False, xycoords='data')
    ax.add_artist(ab)
    
    #Plot surrounding images
    radius = 0.1  #Radius from center for the surrounding images
    for i, angle in enumerate(rots):
    
        theta = np.deg2rad(angle)  
        img = OffsetImage(images[i], zoom=1.9, cmap='turbo')
        ab = AnnotationBbox(img, (theta, radius), frameon=False, xycoords='data')
        ax.add_artist(ab)
        text_radius = radius + 0.05
        if angle != -180.0 : 
            ax.text(theta, text_radius, f'{angle:.1f}°',
            ha='center', va='bottom', fontsize=12, color='black', rotation=0)

    ax.set_rlim(0, radius)
    
    plt.show()
    