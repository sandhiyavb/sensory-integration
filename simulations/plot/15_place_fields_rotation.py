import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

folders = ['../data/conflict/activations','../data/conflict/activations_vector']
file = "classified_fields_both_inputs.csv"

#collect data for plotting
place_fields = {'both' : [], 'vector only' : []}
all_dfs = [] 
for folder, key in zip(folders, place_fields.keys()) : 
    df = pd.read_csv(os.path.join(folder, file))
    df_avg = df.groupby(['run','rotation','unit_type']).size().reset_index(name='Count')
    df_avg = df_avg.groupby(['rotation','unit_type'])['Count'].mean().reset_index(name='average number of place cells')
    
    df_place = df_avg[df_avg['unit_type'] == 'Place'].copy()
    df_place['source'] = key  
    
    all_dfs.append(df_place)  

#Combine all results into a single DataFrame 
df_result = pd.concat(all_dfs, ignore_index=True)
custom_palette = {
    'both': '#BFAF80',
    'vector only': '#454545'
}   
plt.figure(figsize=(6, 4),dpi=300)

sns.lineplot(df_result, x='rotation',y='average number of place cells', hue='source', palette=custom_palette,    # Outline color
    marker='o' ,alpha=1)
sns.despine()
plt.ylim(0)   # y-axis starts at 0
plt.gca().xaxis.set_major_locator(MultipleLocator(60))

