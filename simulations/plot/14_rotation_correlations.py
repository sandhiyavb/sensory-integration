import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import math
from scipy.ndimage import center_of_mass

folders = ['../data/conflict/activations', '../data/conflict/activations_vector']
legend = ['both','vector only']
idxs = [0,1]
runs = np.arange(1,16,1)
rots = np.arange(-180,210,30)

file = 'classified_fields_both_inputs.csv'
#get all place fields for a specific run
all_df_longs = []
for idx in idxs : 
    df = pd.read_csv(os.path.join(folders[idx],file))
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
    
    #choose a run to visualize
    angles_all_runs = []
    indices_list = []
    runs = df_common['run']
    
    
    for run in runs : 
        common_runs = df_common[df_common['run']==run]['common_units'].to_list()
        indices_list.append(common_runs[0])
        #load activations file for this run
        activations = [os.path.join(folders[idx], f'run_{run}','activations',f'activations_{rot}.npy') for rot in rots]
        #maps of place like units
        activations_all = [np.squeeze(np.load(activation))[:,:,common_runs[0]] for activation in activations]        
        activations_mean = [np.reshape(np.mean(field, axis=0), (25,25,-1)) for field in activations_all]
        ref_idx = 6
        ref_array = activations_mean[ref_idx]
        angles_per_image = []
        n_fields = activations_mean[0].shape[-1]
        center = np.array([12,12]) 
        for i in range(n_fields) : 
            ref_img = ref_array[:,:,i]
            ref_max_pos = np.unravel_index(np.argmax(ref_img), ref_img.shape)
            ref_max_pos = np.array(ref_max_pos[::-1]) 
            ref_max_pos = center_of_mass(ref_img)
            vec_ref = ref_max_pos - center
            angles = []
            for arr in activations_mean : 
                img = arr[:,:,i]
                max_pos = np.unravel_index(np.argmax(img), img.shape)
                max_pos = np.array(max_pos[::-1])
                max_pos = center_of_mass(img)
                vec_curr = max_pos - center
                angle_ref = math.atan2(vec_ref[1], vec_ref[0])
                angle_curr = math.atan2(vec_curr[1], vec_curr[0])
                angle_deg = math.degrees(angle_curr - angle_ref)
                angle_deg = ((angle_deg + 180) % 360) - 180
                

                angles.append(angle_deg)
        
            angles_per_image.append(angles)
            angles_array = np.array(angles_per_image)
        angles_all_runs.append(angles_array)        
    
    dfs = []
    
    for run_id, (array, index_array) in enumerate(zip(angles_all_runs, indices_list)):
        df = pd.DataFrame(array, columns=rots)
        df['run'] = run_id + 1
        df['index'] = index_array  # use provided indices
        dfs.append(df)
    
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Optional: reorder columns
    cols = ['run', 'index'] + list(rots)
    final_df = final_df[cols]
    angle_correlations = final_df.drop(['run','index'],axis=1)
    
    df_long = angle_correlations.melt(var_name='angle', value_name='value')
    df_long['angle'] = pd.to_numeric(df_long['angle'])
    
    df_long['source'] = legend[idx]  # Tag the source (e.g., 'activations' or 'activations_vector')
    all_df_longs.append(df_long) 
    # Now plot
combined_df = pd.concat(all_df_longs, ignore_index=True)
custom_palette = {
    'both': '#BFAF80',
    'vector only': '#454545'
}
# Plot with hue based on source
plt.figure(figsize=(10, 6),dpi=300)
sns.stripplot(data=combined_df, x='angle', y='value', hue='source',palette=custom_palette)
#sns.catplot(
#    data=combined_df, x="angle", y="value", hue="source",
#    kind="violin", bw_adjust=.5, cut=0, split=False,
#)
sns.despine()
plt.xlabel('input rotation',fontsize=12)
plt.ylabel('average rotation place field center',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.legend(title='Source', fontsize=8, title_fontsize=8)
plt.grid(False)
plt.show()
