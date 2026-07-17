
# SURF Results Analysis
# This script processes and analyzes SURF (Speeded-Up Robust Features) results from a dataset.
# It was originally a Jupyter Notebook and has been converted into a Python script.

#!/usr/bin/env python
# coding: utf-8

# # Analysis of results from SURF model
# ## keypoint-moseq

# ## 0. Configuring paths and loading results

# In[1]:


ELS_color = '#c37ba0' 
control_color = '#f9c74f'

import os
import time
import jax
import jax.numpy as jnp
import keypoint_moseq as kpms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap.umap_ import UMAP
import hdbscan
from mpl_toolkits.mplot3d import Axes3D

print("JAX devices:", jax.devices())

VIDEO_DIR = "/mnt/g/Big_computer/JEN_STORAGE/equipo_project_data/Videos_all"
data_dir = "/mnt/g/Big_computer/JEN_STORAGE/equipo_project_data/"
project_dir = "/mnt/h/antonio/keypoint_moseq_project/code/equipo_project"

model_name = "2025_01_24-16_44_21"
#moseq_surf_results_path = "G:\\Big_computer\\JEN_STORAGE\\equipo_project_data\\SURF_results\\2025_01_24-16_44_21"



# ## 1. Labeling similar behavioral syllables

# In[2]:


# already done
#kpms.interactive_group_setting(project_dir, model_name)
#
moseq_df = kpms.compute_moseq_df(project_dir, model_name, smooth_heading=True) 
moseq_df.keys()


# In[3]:


FPS = 25 # JEN CHECK THIS
stats_df = kpms.compute_stats_df(
    project_dir,
    model_name,
    moseq_df, 
    min_frequency=0.005,       # threshold frequency for including a syllable in the dataframe
    groupby=['group', 'name'], # column(s) to group the dataframe by
    fps=25)                    # frame rate of the video from which keypoints were inferred

stats_df.keys()


# ## save dataframe and stats (optional)

# In[7]:


#save_dir = os.path.join(project_dir, model_name) # directory to save the moseq_df dataframe
#moseq_df.to_csv(os.path.join(save_dir, 'moseq_df.csv'), index=False)
#print('Saved `moseq_df` dataframe to', save_dir)

# save stats_df
#save_dir = os.path.join(project_dir, model_name)
#stats_df.to_csv(os.path.join(save_dir, 'stats_df'), index=False)
#print('Saved `stats_df` dataframe to', save_dir)

## Label syllables
kpms.label_syllables(project_dir, model_name, moseq_df) 


# ## Overlay points on video

# In[ ]:


#import os
#import h5py
#import numpy as np
#import jax.numpy as jnp
#from jax_moseq.utils import unbatch
#from jax_moseq.models.keypoint_slds import estimate_coordinates
#model, _, metadata, _ = kpms.load_checkpoint(project_dir, model_name)
# compute the estimated coordinates
#Y_est = estimate_coordinates(
#    jnp.array(model['states']['x']),
#    jnp.array(model['states']['v']),
#    jnp.array(model['states']['h']),
#    jnp.array(model['params']['Cd'])
#)
# generate a dictionary with reconstructed coordinates for each recording
#coordinates_est = unbatch(Y_est, *metadata)
#config = lambda: kpms.load_config(project_dir)
#keypoint_data_path = 'dlc_project/videos' # can be a file, a directory, or a list of files
#keypoint_data_path = "/mnt/g/Big_computer/JEN_STORAGE/equipo_project_data/CSVs_all_filtered"
#coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, 'deeplabcut')
#recording_name = 'Animal 34_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000filtered'
#video_path = '/mnt/g/Big_computer/JEN_STORAGE/equipo_project_data/Videos_all/Animal 34_4.avi'
#output_path = os.path.splitext(video_path)[0]+'_reconstructed_keypoints.mp4'
#start_frame, end_frame = 0, 3600
#start_frame, end_frame = 0, 500
#kpms.overlay_keypoints_on_video(
#    video_path,
#    coordinates_est[recording_name],
#    skeleton = config()['skeleton'],
#    bodyparts = config()['use_bodyparts'],
#    output_path = output_path,
#    frames = range(start_frame, end_frame)
#)


# ## Plot stats

# In[4]:


## Plot stats
# ANTONIO TODO plot these using scatters and violin so we see if there are clusters
# within the groups which are later identified using clustering with the behavioral transition matrix
kpms.plot_syll_stats_with_sem(
    stats_df, project_dir, model_name,
    plot_sig=True,    # whether to mark statistical significance with a star
    thresh=0.05,      # significance threshold
    stat='frequency', # statistic to be plotted (e.g. 'duration' or 'velocity_px_s_mean')
    order='stat',     # order syllables by overall frequency ("stat") or degree of difference ("diff")
    ctrl_group='Control',   # name of the control group for statistical testing
    exp_group='ELS',    # name of the experimental group for statistical testing
    #figsize=(8, 4),   # figure size    
    groups=stats_df['group'].unique(), # groups to be plotted
);


# ## Retrieving freezing predictions from Jen

# In[5]:


freezing_dir = r"/mnt/h/antonio/keypoint_moseq_project/code/equipo_project/Freezing_predictions_light"

freezing_data = {}
for fname in os.listdir(freezing_dir):
    if fname.endswith("_freezing_predictions_only.csv"):
        
        # Full path to CSV
        full_path = os.path.join(freezing_dir, fname)
        
        # Remove "_freezing_predictions_only.csv" at the end
        base_name = fname.replace("_freezing_predictions_only.csv", "")
        
        # Load CSV into a pandas DataFrame
        df_freezing = pd.read_csv(full_path)
        
        # Add to dictionary
        freezing_data[base_name] = df_freezing

print("Loaded keys:", list(freezing_data.keys()))


# ## Plotting syllables for a specific mouse

# In[6]:


unique_names = moseq_df['name'].unique()

def parse_moseq_name(moseq_name):
    """
    A simple parser that removes the 'DLC...' suffix or any appended text
    after the main 'Animal X_Y' portion.
    Adjust logic as needed for your naming conventions.
    """
    if 'DLC' in moseq_name:
        return moseq_name.split('DLC')[0].rstrip('_')
    return moseq_name

max_plots = 15
plots = 0

for animal_name in unique_names:
    if plots >= max_plots:
        break
    plots += 1

    # Filter the MoSeq dataframe to get data for this animal
    df_animal = moseq_df[moseq_df['name'] == animal_name]

    # Derive the "base name" to look up in `freezing_data`
    base_name = parse_moseq_name(animal_name)
    if base_name not in freezing_data:
        print(f"No freezing CSV for {base_name} - skipping shading.")
        df_freeze = None
    else:
        df_freeze = freezing_data[base_name]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(df_animal['frame_index'], df_animal['syllable'], 
               s=1, c='black', label='All syllables')

    # Highlight specific syllables (0 and 28) with different colors
    #special_syllables = [0, 28]
    special_syllables = [0, 1, 13, 28]
    # special_syllables = [0]

    
    colors = ['blue', 'red', 'green','brown' ]
    markers = ['x', 'x', 'x', 'x']
    for syllable, color, marker in zip(special_syllables, colors, markers):
        df_syll = df_animal[df_animal['syllable'] == syllable]
        ax.scatter(df_syll['frame_index'], df_syll['syllable'],
                   s=50, c=color, marker=marker, label=f'Syllable {syllable}')

    # -------------------------------------------------------
    # Shade freezing frames if we have a matching CSV
    # -------------------------------------------------------
    # df_freeze expected to have columns: 'Unnamed: 0', 'Freezing_Jen_0-125_threshold'
    if df_freeze is not None and 'Freezing_Jen_0-125_threshold' in df_freeze.columns:
        # Get all frame indices where "Freezing_Jen_0-125_threshold" == 1
        freeze_indices = df_freeze.loc[
            df_freeze['Freezing_Jen_0-125_threshold'] == 1, 'Unnamed: 0'
        ].to_numpy()

        if len(freeze_indices) > 0:
            # Identify consecutive runs (start->end) of freezing
            runs = []
            start = freeze_indices[0]
            end = start

            for fidx in freeze_indices[1:]:
                if fidx == end + 1:
                    # continue run
                    end = fidx
                else:
                    # close the old run, start a new run
                    runs.append((start, end))
                    start = fidx
                    end = fidx
            # Add the last run
            runs.append((start, end))

            # Shade each run with axvspan (in blue, low alpha)
            for (run_start, run_end) in runs:
                ax.axvspan(run_start, run_end + 1, color='blue', alpha=0.1, label='_nolegend_')

    # Final labeling
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Syllable')
    ax.set_title(f'Syllable Over Time: {animal_name}')
    ax.legend()
    plt.show()


# ## Overlap between selected freezing syllables and ground truth freezing

# In[7]:


selected_syllables = [0, 1, 13, 28]

freezing_overlap = []

# Precompute freezing indices and syllable frames for all animals
animal_freezing_data = {
    animal_name: freezing_data[parse_moseq_name(animal_name)].loc[
        freezing_data[parse_moseq_name(animal_name)]['Freezing_Jen_0-125_threshold'] == 1, 'Unnamed: 0'
    ].to_numpy()
    if parse_moseq_name(animal_name) in freezing_data else None
    for animal_name in moseq_df['name'].unique()}

animal_syllable_data = {
    animal_name: moseq_df[moseq_df['name'] == animal_name][['syllable', 'frame_index']].groupby('syllable')['frame_index'].apply(set).to_dict()
    for animal_name in moseq_df['name'].unique()}

# Precompute group data
animal_groups = {
    animal_name: moseq_df.loc[moseq_df['name'] == animal_name, 'group'].values[0]
    for animal_name in moseq_df['name'].unique()}

# Loop through each animal and calculate overlaps
for animal_name, freeze_indices in animal_freezing_data.items():
    if freeze_indices is None:  # Skip animals with no freezing data
        freezing_overlap.append({
            'animal': animal_name,
            'group': animal_groups[animal_name],
            'total_freeze_frames': 0,
            'overlap_frames': 0,
            'percent_overlap': 0.0,
            'note': 'No freezing data'})
        continue
    total_freeze_frames = len(freeze_indices)
    if total_freeze_frames == 0:
        freezing_overlap.append({
            'animal': animal_name,
            'group': animal_groups[animal_name],
            'total_freeze_frames': 0,
            'overlap_frames': 0,
            'percent_overlap': 0.0,
            'note': 'No freezing frames'})
        continue
    # Get frames for selected syllables
    syllable_frames = set()
    for syllable in selected_syllables:
        if syllable in animal_syllable_data[animal_name]:
            syllable_frames.update(animal_syllable_data[animal_name][syllable])
            
    overlap = set(freeze_indices).intersection(syllable_frames)
    overlap_count = len(overlap)
    overlap_percent = (overlap_count / total_freeze_frames) * 100
    
    freezing_overlap.append({
        'animal': animal_name,
        'group': animal_groups[animal_name],
        'total_freeze_frames': total_freeze_frames,
        'overlap_frames': overlap_count,
        'percent_overlap': overlap_percent,
        'note': ''})

# Extract percent overlaps and groups
percent_overlaps = [entry['percent_overlap'] for entry in freezing_overlap]
groups = [entry['group'] for entry in freezing_overlap]

# Compute overall mean and std
mean_all = np.mean(percent_overlaps)
std_all = np.std(percent_overlaps)

# Separate overlaps by group
els_overlaps = [entry['percent_overlap'] for entry in freezing_overlap if entry['group'] == 'ELS']
control_overlaps = [entry['percent_overlap'] for entry in freezing_overlap if entry['group'] == 'Control']

# Compute mean and std for each group
mean_els = np.mean(els_overlaps) if els_overlaps else 0  # Avoid errors if no ELS data
std_els = np.std(els_overlaps) if els_overlaps else 0
mean_control = np.mean(control_overlaps) if control_overlaps else 0
std_control = np.std(control_overlaps) if control_overlaps else 0

print(f"Overall Percent Overlap: {mean_all:.2f} ± {std_all:.2f}%")
print(f"ELS Percent Overlap: {mean_els:.2f} ± {std_els:.2f}%")
print(f"CONTROL Percent Overlap: {mean_control:.2f} ± {std_control:.2f}%")


# In[8]:


### ----> Create a bar plot for percent overlap
# Extract animals, groups, and percent overlaps
animals = [entry['animal'][7:12] for entry in freezing_overlap]
groups = [entry['group'] for entry in freezing_overlap]
percent_overlaps = [entry['percent_overlap'] for entry in freezing_overlap]
data = list(zip(animals, groups, percent_overlaps))
data_sorted = sorted(data, key=lambda x: x[1] == 'Control')  # 'ELS' evaluates to False, 'Control' to True

animals_sorted, groups_sorted, percent_overlaps_sorted = zip(*data_sorted)
colors_sorted = [ELS_color if group == 'ELS' else control_color for group in groups_sorted]
plt.figure(figsize=(14, 6), dpi=300)
plt.bar(animals_sorted, percent_overlaps_sorted, color=colors_sorted, edgecolor='white', width=1)
plt.xlabel('Animal', fontsize=12)
plt.ylabel('Percent Overlap', fontsize=12)
plt.title('Overlap Between Syllable group and Ground truth Freezing', fontsize=16)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(rotation=90, ha='center', fontsize=10)
plt.tight_layout()
plt.show()


# ## Freezing overlap per syllable for all syllables

# In[9]:


# freezing_overlap_per_syllable = []

# # Precompute freezing indices and syllable data for each animal to avoid redundant filtering
# animal_freezing_data = {
#     entry['animal']: {
#         'group': entry['group'],
#         'freeze_indices': freezing_data[parse_moseq_name(entry['animal'])].loc[
#             freezing_data[parse_moseq_name(entry['animal'])]['Freezing_Jen_0-125_threshold'] == 1, 'Unnamed: 0'
#         ].to_numpy()}
#     for entry in freezing_overlap}

# animal_syllable_data = {
#     animal: moseq_df[moseq_df['name'] == animal].groupby('syllable')['frame_index'].apply(set).to_dict()
#     for animal in moseq_df['name'].unique()}

# # Compute freezing overlap per syllable
# for animal_name, data in animal_freezing_data.items():
#     group = data['group']
#     freeze_indices = set(data['freeze_indices'])  # Convert to set once for efficient lookups
#     total_freeze_frames = len(freeze_indices)

#     if total_freeze_frames > 0:  # Skip if no freezing frames
#         for syllable, syllable_frames in animal_syllable_data[animal_name].items():
#             overlap = freeze_indices.intersection(syllable_frames)
#             freezing_overlap_per_syllable.append({
#                 'animal': animal_name,
#                 'syllable': syllable,
#                 'group': group,
#                 'total_freeze_frames': total_freeze_frames,
#                 'overlap_frames': len(overlap),
#                 'percent_overlap': (len(overlap) / total_freeze_frames) * 100
#             })



# In[10]:


animal_freezing_data.keys()


# In[11]:


# Precompute freezing indices and syllable data for each animal to avoid redundant filtering
animal_freezing_data = {
     entry['animal']: {
         'group': entry['group'],
         'freeze_indices': freezing_data[parse_moseq_name(entry['animal'])].loc[
             freezing_data[parse_moseq_name(entry['animal'])]['Freezing_Jen_0-125_threshold'] == 1, 'Unnamed: 0'
         ].to_numpy()}
     for entry in freezing_overlap}

animal_syllable_data = {
     animal: moseq_df[moseq_df['name'] == animal].groupby('syllable')['frame_index'].apply(set).to_dict()
     for animal in moseq_df['name'].unique()}


freezing_overlap_per_syllable = []

for animal_name, data in animal_freezing_data.items():
    group = data['group']
    freeze_indices = set(data['freeze_indices'])  # frames where freezing == 1
    total_freeze_frames = len(freeze_indices)

    # Skip animals with no freezing frames
    if total_freeze_frames == 0:
        continue

    # Get syllable -> set_of_frames for this animal
    if animal_name not in animal_syllable_data:
        continue
    syll_dict = animal_syllable_data[animal_name]

    for syllable, syll_frames in syll_dict.items():
        # Overlap: frames that are both 'freezing' and labeled with this syllable
        overlap = freeze_indices.intersection(syll_frames)
        overlap_count = len(overlap)

        total_syllable_frames = len(syll_frames)

        # -- "Recall": % of all freezing frames covered by this syllable
        #    i.e. overlap / total_freeze_frames
        recall = 0.0
        if total_freeze_frames > 0:
            recall = (overlap_count / total_freeze_frames)

        # -- "Precision": % of syllable frames that are freezing
        #    i.e. overlap / total_syllable_frames
        precision = 0.0
        if total_syllable_frames > 0:
            precision = (overlap_count / total_syllable_frames)

        # -- [Optional] F1 score = harmonic mean of precision & recall
        f1 = 0.0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        freezing_overlap_per_syllable.append({
            'animal': animal_name,
            'syllable': syllable,
            'group': group,
            'total_freeze_frames': total_freeze_frames,
            'total_syllable_frames': total_syllable_frames,
            'overlap_frames': overlap_count,
            'percent_freezing_covered': recall * 100.0,     # ~ "recall %"
            'percent_syllable_in_freezing': precision * 100.0,  # ~ "precision %"
            'f1_score': f1
        })

# Convert the list of dicts into a DataFrame for analysis
import pandas as pd
metrics_df = pd.DataFrame(freezing_overlap_per_syllable)
print(metrics_df.head(15))


# In[13]:


import numpy as np
import matplotlib.pyplot as plt


def plot_top_90(
    data, measure_key, 
    ylabel,            # e.g. "Mean Recall (%)"
    dashed_line_value, # e.g. 5 or 0.5
    dashed_line_label, # e.g. "5% Threshold" or "F1=0.5"
    title
):
    """
    1) Aggregate 'measure_key' by syllable & group
    2) Sort descending by mean_all
    3) Compute top 90% coverage
    4) Bar plot
    """
    # 1) Aggregate
    syllable_stats = []
    all_syllables = list({entry['syllable'] for entry in data})
    
    for syl in all_syllables:
        vals_all = [entry[measure_key] for entry in data if entry['syllable'] == syl]
        vals_els = [entry[measure_key] for entry in data if entry['syllable'] == syl and entry['group'] == 'ELS']
        vals_ctrl= [entry[measure_key] for entry in data if entry['syllable'] == syl and entry['group'] == 'Control']
        
        mean_all = np.mean(vals_all) if vals_all else 0
        mean_els = np.mean(vals_els) if vals_els else 0
        mean_ctrl= np.mean(vals_ctrl) if vals_ctrl else 0

        syllable_stats.append({
            'syllable': syl,
            'mean_all': mean_all,
            'mean_els': mean_els,
            'mean_control': mean_ctrl
        })
    
    # 2) Sort descending by mean_all
    syllable_stats.sort(key=lambda x: x['mean_all'], reverse=True)
    
    # 3) Compute top 90% coverage
    total_val = sum(s['mean_all'] for s in syllable_stats)
    cumulative_contribution = 0
    selected_syllables = []
    for stat in syllable_stats:
        if total_val > 0:
            cumulative_contribution += (stat['mean_all'] / total_val) * 100
        selected_syllables.append(stat)
        if cumulative_contribution >= 90:
            break
    
    # 4) Prepare data for plotting
    x_labels     = [f"S{stat['syllable']}" for stat in selected_syllables]
    mean_all     = [stat['mean_all'] for stat in selected_syllables]
    mean_els     = [stat['mean_els']  for stat in selected_syllables]
    mean_control = [stat['mean_control'] for stat in selected_syllables]
    
    x = np.arange(len(x_labels))
    width = 0.2
    
    # Bar plot
    plt.figure(figsize=(12, 6), dpi=300)
    plt.bar(x - width, mean_all, width, label='All Groups', color='black', edgecolor='black')
    plt.bar(x,        mean_els, width, label='ELS',        color='#c37ba0', edgecolor='black')
    plt.bar(x + width,mean_control, width, label='Control', color='#f9c74f', edgecolor='black')
    
    plt.axhline(y=dashed_line_value, color='gray', linestyle='--', linewidth=1.5, label=dashed_line_label)
    
    plt.xlabel('Syllable', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title + ' [Top 90%]', fontsize=14)
    plt.xticks(x, x_labels, rotation=90, fontsize=10)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------
# Now we call plot_top_90 for each measure (recall, precision, f1)
# ------------------------------------------------------------------------
# 1) Recall (coverage of freezing frames)
plot_top_90(
    data=freezing_overlap_per_syllable,
    measure_key='percent_freezing_covered',
    ylabel='Mean Recall (%)',
    dashed_line_value=5,              # show a 5% threshold
    dashed_line_label='5% Threshold',
    title='Mean Recall (coverage of freezing frames)'
)

# 2) Precision (syllable frames that are freezing)
plot_top_90(
    data=freezing_overlap_per_syllable,
    measure_key='percent_syllable_in_freezing',
    ylabel='Mean Precision (%)',
    dashed_line_value=5,  # same 5% threshold as reference
    dashed_line_label='5% Threshold',
    title='Mean Precision (syllable frames that are freezing)'
)

# 3) F1 Score
plot_top_90(
    data=freezing_overlap_per_syllable,
    measure_key='f1_score',
    ylabel='Mean F1 Score',
    dashed_line_value=0.5,
    dashed_line_label='F1=0.5',
    title='Mean F1 Score (Precision/Recall Harmonic Mean)'
)


# ## Total frames per syllable

# In[18]:


# 1) Count how many rows (frames) each syllable has
syllable_counts = moseq_df['syllable'].value_counts().sort_index()

# 2) Prepare x (syllable IDs) and y (counts)
x = syllable_counts.index
counts = syllable_counts.values

# 3) Compute percentage
total_frames = counts.sum()
percentages = (counts / total_frames) * 100

# Print a quick check to confirm it sums to ~100%
print(f"Sum of all syllable percentages: {percentages.sum():.2f}%")

plt.figure(figsize=(16, 5), dpi=300)
plt.bar(x, percentages, color='skyblue', edgecolor='black')
plt.xlabel("Syllable ID", fontsize=12)
plt.ylabel("Percent of Total Frames", fontsize=12)
plt.title("Percentage of Total Frames per Syllable", fontsize=14)
plt.xticks(x, [f"S{val}" for val in x], rotation=90)

ymax = percentages.max()
# Create an array from 0 to (rounded-up) maximum, step=1
ytick_vals = np.arange(0, int(np.ceil(ymax)) + 1, 1)
plt.yticks(ytick_vals)

plt.tight_layout()
plt.show()



# In[19]:


import numpy as np
import matplotlib.pyplot as plt

# 1) Count how many rows (frames) each syllable has, then sort descending by count
syllable_counts = moseq_df['syllable'].value_counts().sort_values(ascending=False)

# 2) Prepare x (syllable IDs) and y (counts)
x = syllable_counts.index
counts = syllable_counts.values

# 3) Compute percentage
total_frames = counts.sum()
percentages = (counts / total_frames) * 100

print(f"Sum of all syllable percentages: {percentages.sum():.2f}%")

plt.figure(figsize=(16, 5), dpi=300)
plt.bar(range(len(x)), percentages, color='skyblue', edgecolor='black')  # x-axis is just 0..N-1
plt.xlabel("Syllable (sorted by count)", fontsize=12)
plt.ylabel("Percent of Total Frames", fontsize=12)
plt.title("Percentage of Total Frames per Syllable (Descending Order)", fontsize=14)

# Set x-ticks to syllable indices with appropriate labels
plt.xticks(
    range(len(x)), 
    [f"S{val}" for val in x], 
    rotation=90
)

# Force y-ticks at every integer % from 0 to the rounded-up maximum
ymax = percentages.max()
ytick_vals = np.arange(0, int(np.ceil(ymax)) + 1, 1)
plt.yticks(ytick_vals)

plt.tight_layout()
plt.show()


# # TODO
# 
# ### 1. Precision and recall per syllable (how much % of that syllable is freezing), similar plot as overlap with freezing (see figure above). That helps select the syllables better
# ### 2. Cluster UMAP and see if the freezing syllables are together
# ### 3. Dendrogram puts together freezing syllables?
# ### 4. Once this works, put all these syllables as one, then do the result statistics
# 
# ###
# 

# In[254]:


0 to fps * 3 * 60 # exploration
3  to fps * 3.30 * 60 # tone
  to fps * 3.30 * 60 # tone






# ## Stats over time

# In[55]:


# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:59:25 2024

@author: jsangui
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, mannwhitneyu

# Define new bin size and FPS
bin_size_frames = 750
bin_size_frames = 25
fps = 25

# Create a new column for time bins based on frame_index
moseq_df['time_bin'] = (moseq_df['frame_index'] // bin_size_frames)

syllables_to_merge = [0, 1, 40, 28]
#syllables_to_merge = [34]

# Create a new column 'syllable_combined' where syllables 0 and 35 are replaced with 999
moseq_df['syllable_combined'] = moseq_df['syllable'].apply(lambda x: 999 if x in syllables_to_merge else x)

# Group by 'group', 'time_bin', 'syllable_combined', and 'name', and count occurrences
grouped = moseq_df.groupby(['group', 'time_bin', 'syllable_combined', 'name']).size().reset_index(name='syllable_frame_count')
total_frames_per_bin = moseq_df.groupby(['group', 'time_bin', 'name']).size().reset_index(name='total_frame_count')

# Merge to get both syllable counts and total frame counts in the same DataFrame
merged_df = pd.merge(grouped, total_frames_per_bin, on=['group', 'time_bin', 'name'])

# Calculate the percentage of time each syllable occupies in the time bin
merged_df['percentage_time'] = (merged_df['syllable_frame_count'] / merged_df['total_frame_count']) * 100


# Define the combined syllable of interest
selected_syllable = 999  # The new combined syllable

# Define the colors for each group
group_colors = {
    'Control': '#f9c74f',  # Control group color
    'ELS': '#c37ba0',     # ELS group color
}

# Filter the merged DataFrame to only include the selected syllable
filtered_df = merged_df[merged_df['syllable_combined'] == selected_syllable]

# Determine the maximum time_bin value
max_time_bin = filtered_df['time_bin'].max()

# Create a single plot for both groups
fig, ax = plt.subplots(figsize=(14, 6))

# Create a list to store all individual data points per time_bin
all_percentages = []

# Loop through each group and plot the data
for i, group in enumerate(['Control', 'ELS']):
    # Filter data for the current group
    group_df = filtered_df[filtered_df['group'] == group]

    # Slightly offset x-positions for each group to prevent overlap
    offset = -0.2 if group == 'Control' else 0.2

    # Plot individual data points per time_bin
    for time_bin in sorted(group_df['time_bin'].unique()):
        bin_df = group_df[group_df['time_bin'] == time_bin]
        percentages = bin_df['percentage_time']
        x_positions = np.full_like(percentages, fill_value=time_bin + 1 + offset, dtype=np.double)

        # Add a small random jitter to prevent overlapping points
        x_positions += np.random.uniform(-0.05, 0.05, size=len(percentages))

        # Scatter plot of individual data points
        ax.scatter(x_positions, percentages, color=group_colors[group], alpha=0.7, edgecolor='black', linewidth=0.5, label=group if time_bin == 0 else "")

        # Collect all percentages for max y-value calculation
        all_percentages.extend(zip(x_positions, percentages, [time_bin] * len(percentages)))

# Calculate the maximum y-values per time_bin across both groups
max_y_values = {}
for time_bin in sorted(filtered_df['time_bin'].unique()):
    time_bin_percentages = [y for x, y, tb in all_percentages if tb == time_bin]
    if time_bin_percentages:
        max_y_values[time_bin] = max(time_bin_percentages)
    else:
        max_y_values[time_bin] = 0

# Perform statistical tests per time bin
p_values = {}
for time_bin in sorted(filtered_df['time_bin'].unique()):
    bin_df = filtered_df[filtered_df['time_bin'] == time_bin]
    group1_data = bin_df[bin_df['group'] == 'Control']['percentage_time']
    group2_data = bin_df[bin_df['group'] == 'ELS']['percentage_time']
    n1 = len(group1_data)
    n2 = len(group2_data)
    if n1 >= 3 and n2 >= 3:
        # Perform Mann-Whitney U test
        stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
    else:
        p_value = np.nan  # Not enough data
    p_values[time_bin] = p_value

# Bonferroni correction
alpha = 0.05
num_tests = len(p_values)
adjusted_alpha = alpha / num_tests  # Adjusted significance level
adjusted_p_values = {time_bin: min(p * num_tests, 1.0) if not np.isnan(p) else np.nan for time_bin, p in p_values.items()}

# Determine y_offset based on y-axis range
ymin, ymax = ax.get_ylim()
y_offset = (ymax - ymin) * 0.05  # 5% of y-axis range

# Plot stars for significant adjusted p-values
for time_bin, p_value in adjusted_p_values.items():
    if p_value < alpha:
        x = time_bin + 1  # Adjust x position
        y = max_y_values[time_bin] + y_offset
        ax.text(x, y, '*', ha='center', va='bottom', color='black', fontsize=40)

# Adjust y-axis limit to make sure stars fit in
new_ymax = max(max_y_values.values()) + y_offset * 2
ax.set_ylim(ymin, new_ymax)

# Set titles and labels
ax.set_title(f'Syllable {selected_syllable}', color='#4d4d4d', fontsize=14)
ax.set_xlabel('\nTime (minutes)', color='#4d4d4d', fontsize=13, ha='right', x=1.0)
ax.set_ylabel('Percentage of Time (%)', color='#4d4d4d', fontsize=13)
ax.tick_params(axis='x', colors='#4d4d4d', labelsize=13)
ax.tick_params(axis='y', colors='#4d4d4d', labelsize=13)

# Update x-axis ticks and labels
custom_ticks = [1, 2, 3, 4, 5, 6, 7]
custom_labels = ['1', '2', '3', '4', '5', '6', '7']
ax.set_xticks(custom_ticks)
ax.set_xticklabels(custom_labels)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#4d4d4d')
ax.spines['bottom'].set_color('#4d4d4d')

# Add legend
ax.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()


# In[56]:





# In[58]:


# Define the combined syllable of interest
selected_syllable = 999  # The new combined syllable

# Define the colors for each group
group_colors = {
    'Control': '#f9c74f',  # Control group color
    'ELS': '#c37ba0',     # ELS group color
}

# Filter the merged DataFrame to only include the selected syllable
filtered_df = merged_df[merged_df['syllable_combined'] == selected_syllable]

# Determine the maximum time_bin value
max_time_bin = filtered_df['time_bin'].max()

# Create a single plot for both groups
fig, ax = plt.subplots(figsize=(14, 6))

# Create a list to store all individual data points per time_bin
all_percentages = []

# Loop through each group and plot the data
for i, group in enumerate(['Control', 'ELS']):
    # Filter data for the current group
    group_df = filtered_df[filtered_df['group'] == group]

    # Slightly offset x-positions for each group to prevent overlap
    offset = -0.2 if group == 'Control' else 0.2

    # Plot individual data points per time_bin
    for time_bin in sorted(group_df['time_bin'].unique()):
        bin_df = group_df[group_df['time_bin'] == time_bin]
        percentages = bin_df['percentage_time']
        x_positions = np.full_like(percentages, fill_value=time_bin + 1 + offset, dtype=np.double)

        # Add a small random jitter to prevent overlapping points
        x_positions += np.random.uniform(-0.05, 0.05, size=len(percentages))

        # Scatter plot of individual data points
        ax.scatter(x_positions, percentages, color=group_colors[group], alpha=0.7, edgecolor='black', linewidth=0.5, label=group if time_bin == 0 else "")

        # Collect all percentages for max y-value calculation
        all_percentages.extend(zip(x_positions, percentages, [time_bin] * len(percentages)))

# Calculate the maximum y-values per time_bin across both groups
max_y_values = {}
for time_bin in sorted(filtered_df['time_bin'].unique()):
    time_bin_percentages = [y for x, y, tb in all_percentages if tb == time_bin]
    if time_bin_percentages:
        max_y_values[time_bin] = max(time_bin_percentages)
    else:
        max_y_values[time_bin] = 0

# Perform statistical tests per time bin
p_values = {}
for time_bin in sorted(filtered_df['time_bin'].unique()):
    bin_df = filtered_df[filtered_df['time_bin'] == time_bin]
    group1_data = bin_df[bin_df['group'] == 'Control']['percentage_time']
    group2_data = bin_df[bin_df['group'] == 'ELS']['percentage_time']
    n1 = len(group1_data)
    n2 = len(group2_data)
    if n1 >= 3 and n2 >= 3:
        # Perform Mann-Whitney U test
        stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
    else:
        p_value = np.nan  # Not enough data
    p_values[time_bin] = p_value

# Bonferroni correction
alpha = 0.05
num_tests = len(p_values)
adjusted_alpha = alpha / num_tests  # Adjusted significance level
adjusted_p_values = {time_bin: min(p * num_tests, 1.0) if not np.isnan(p) else np.nan for time_bin, p in p_values.items()}

# Determine y_offset based on y-axis range
ymin, ymax = ax.get_ylim()
y_offset = (ymax - ymin) * 0.05  # 5% of y-axis range

# Plot stars for significant adjusted p-values
for time_bin, p_value in adjusted_p_values.items():
    if p_value < alpha:
        x = time_bin + 1  # Adjust x position
        y = max_y_values[time_bin] + y_offset
        ax.text(x, y, '*', ha='center', va='bottom', color='black', fontsize=40)

# Adjust y-axis limit to make sure stars fit in
new_ymax = max(max_y_values.values()) + y_offset * 2
ax.set_ylim(ymin, new_ymax)

# Set titles and labels
ax.set_title(f'Syllable {selected_syllable}', color='#4d4d4d', fontsize=14)
ax.set_xlabel('\nTime (minutes)', color='#4d4d4d', fontsize=13, ha='right', x=1.0)
ax.set_ylabel('Percentage of Time (%)', color='#4d4d4d', fontsize=13)
ax.tick_params(axis='x', colors='#4d4d4d', labelsize=13)
ax.tick_params(axis='y', colors='#4d4d4d', labelsize=13)

# Update x-axis ticks and labels
custom_ticks = [1, 2, 3, 4, 5, 6, 7]
custom_labels = ['1', '2', '3', '4', '5', '6', '7']
ax.set_xticks(custom_ticks)
ax.set_xticklabels(custom_labels)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#4d4d4d')
ax.spines['bottom'].set_color('#4d4d4d')

# Add legend
ax.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()


# ## Next stuff

# In[ ]:





# In[54]:


import numpy as np

# Identify all possible syllables in your dataset for one-hot / n-gram
unique_sylls = np.sort(moseq_df['syllable'].unique())
syll_to_idx = {s: i for i, s in enumerate(unique_sylls)}
n_sylls = len(unique_sylls)

def encode_mouse_syllables(syl_array, mode='raw', ngram_size=5):
    """
    Convert a single mouse's syllable time-series (1D array) into 
    a new representation (raw, one-hot, or n-gram).
    
    Parameters
    ----------
    syl_array : 1D numpy array of shape (n_frames,)
        The per-frame syllables for one mouse.
    mode : str, one of {'raw', 'one-hot', 'ngram'}
        - 'raw': Return the raw syllable IDs (shape ~ (n_frames,)).
        - 'one-hot': Return a (n_frames, n_sylls) array with one-hot encoding 
                     for each frame. (Potentially huge!)
        - 'ngram': Create an array of shape (n_frames - ngram_size + 1, ngram_size),
                   storing consecutive n-grams of integer syllable IDs.
                   (You could also one-hot each n-gram if desired.)
    ngram_size : int
        Size of the consecutive window for 'ngram' mode.

    Returns
    -------
    numpy array
        Encoded representation. Shape depends on `mode`:
          * 'raw' -> (n_frames,)
          * 'one-hot' -> (n_frames, n_sylls)
          * 'ngram' -> (n_frames - ngram_size + 1, ngram_size)
    """

    if mode == 'raw':
        return syl_array

    elif mode == 'one-hot':
        # Each frame -> one-hot vector in [n_sylls]
        n_frames = len(syl_array)
        # Build array of zeros
        encoded = np.zeros((n_frames, n_sylls), dtype=np.float32)
        # Map each syllable ID to its column index
        idx_array = np.array([syll_to_idx[syl] for syl in syl_array])
        rows = np.arange(n_frames)
        encoded[rows, idx_array] = 1.0
        return encoded

    elif mode == 'ngram':
        n_frames = len(syl_array)
        out_length = n_frames - ngram_size + 1
        if out_length < 1:
            # Not enough frames for even one n-gram
            return np.array([])
        # Build array of shape (out_length, ngram_size)
        # Each row has consecutive syllable IDs
        encoded = np.zeros((out_length, ngram_size), dtype=int)
        for start_idx in range(out_length):
            encoded[start_idx, :] = syl_array[start_idx : start_idx + ngram_size]
        return encoded

    else:
        raise ValueError(f"Unknown mode: {mode}")

# 1) Gather each mouse's syllable time-series into a single representation
unique_names = moseq_df['name'].unique()
X_list = []
animal_list = []

# Choose how you want to encode: 'raw', 'one-hot', or 'ngram'
encoding_mode = 'one-hot' 
ngram_size = 10  # used only if encoding_mode='ngram'

for animal_name in unique_names:
    # Extract all frames for this animal, sorted by frame_index if desired
    df_animal = moseq_df[moseq_df['name'] == animal_name].sort_values('frame_index')

    # Convert the syllable column to a 1D numpy array
    syl_array = df_animal['syllable'].to_numpy()
    # Encode
    encoded_rep = encode_mouse_syllables(syl_array, mode=encoding_mode, ngram_size=ngram_size)
    
    X_list.append(encoded_rep)
    animal_list.append(animal_name)

# Now X_list[i] is the encoded representation for mouse i
# If 'raw', shape is (n_frames,); if 'one-hot', shape is (n_frames, n_sylls);
# if 'ngram', shape is (n_frames - ngram_size + 1, ngram_size).

print(f"Number of mice: {len(X_list)}")
print(f"Example shape of the first mouse: {X_list[0].shape}")


# In[203]:


# 1) Flatten each mouse's 2D array into a single 1D vector
#    So each mouse becomes one row (sample) in the final array.
X_flattened = [x.reshape(-1) for x in X_list]  # each x is shape (11241, 10), -> shape (112410,)

# 2) Stack into a final matrix: shape (n_mice, flattened_dim)
X = np.stack(X_flattened, axis=0)
print("Final shape of X:", X.shape)


# In[204]:


# 3) Fit a full PCA to find how many components reach 80% variance
pca_full = PCA()
pca_full.fit(X)  # Fit on all dimensions
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_comps_80 = np.searchsorted(cumulative_variance, 0.70) + 1
print(f"Number of PCA components for >=80% variance: {n_comps_80}")


# In[205]:


## 4) Refit PCA using that many components
pca_80 = PCA(n_components=n_comps_80)
X_pca = pca_80.fit_transform(X)  # shape: [n_mice, n_comps_80]
print("X_pca shape:", X_pca.shape)

# 5) UMAP from the PCA-reduced data -> 3D
reducer = umap.UMAP(n_components=5, n_neighbors=10, min_dist=0.1)
X_umap = reducer.fit_transform(X_pca)  # shape: [n_mice, 3]
print("X_umap shape:", X_umap.shape)


# In[206]:


# 6) HDBSCAN cluster
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
labels = clusterer.fit_predict(X_umap)
print("Cluster labels:", labels)


# In[207]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Build a quick lookup dict from animal_name -> group
# (Assuming moseq_df has columns ["name", "group", ...])
name2group = {}
for a in moseq_df['name'].unique():
    g = moseq_df.loc[moseq_df['name'] == a, 'group'].unique()
    # If an animal has a single group, use that
    # If multiple or none, handle accordingly
    name2group[a] = g[0] if len(g) > 0 else "Unknown"

fig = plt.figure(figsize=(9, 6), dpi=300)
ax = fig.add_subplot(111, projection='3d', facecolor='white')

unique_labels = np.unique(labels)
cmap = plt.cm.get_cmap("Set2", len(unique_labels))

for lbl in unique_labels:
    mask = (labels == lbl)
    if lbl == -1:
        # -1 = noise/outlier
        color = "gray"
        lbl_str = "Outlier"
    else:
        color = cmap(lbl)
        lbl_str = f"Cluster {lbl}"

    ax.scatter(
        X_umap[mask, 0],
        X_umap[mask, 1],
        X_umap[mask, 2],
        c=[color],
        s=60,
        label=lbl_str
    )

# ----------------------------------------------------------------------
#  Find & plot the most central point of each cluster as a black 'X'
# ----------------------------------------------------------------------
for lbl in unique_labels:
    if lbl == -1:
        continue  # skip outlier cluster if desired

    # Indices for data points in this cluster
    cluster_idx = np.where(labels == lbl)[0]
    cluster_points = X_umap[cluster_idx]  # shape (n_points_in_cluster, 3)

    # Compute centroid (mean of all points in cluster)
    centroid_3d = cluster_points.mean(axis=0)

    # Find the single data point closest to the centroid
    distances = np.linalg.norm(cluster_points - centroid_3d, axis=1)
    min_idx_local = np.argmin(distances)
    min_idx_global = cluster_idx[min_idx_local]

    # Plot that "most central" data point with a black X
    ax.scatter(
        X_umap[min_idx_global, 0],
        X_umap[min_idx_global, 1],
        X_umap[min_idx_global, 2],
        marker='X',
        c='black',
        s=40,
        edgecolors='black'
    )

    # Print info on this central data point
    central_animal = animal_list[min_idx_global]
    group_type = name2group.get(central_animal, "Unknown")
    print(f"Cluster {lbl}: center index={min_idx_global}, animal={central_animal}, group={group_type}")

ax.grid(False)
ax.set_xlabel("UMAP-1", labelpad=15)
ax.set_ylabel("UMAP-2", labelpad=15)
ax.set_zlabel("UMAP-3", labelpad=15)
ax.set_title("Syllables UMAP projection", pad=20)
ax.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.show()

# Print or inspect cluster assignments
results_df = pd.DataFrame({
    "animal": animal_list,
    "cluster": labels
})


# In[211]:


# List the specific three animals you want to plot
unique_names_selected = ([
    "Animal_129_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000filtered",  # Example 1
    "Animal_69_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000filtered",  # Example 2
    "Animal_54_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000filtered"   # Example 3
])

unique_names_selected = ([
    "Animal_128_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000filtered",  # ELS
    "Animal_56_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000filtered",  # ELS
    "Animal 34_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000filtered"   # Control
])

unique_names_selected = ([
    "Animal2_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000filtered",  # ELS
    "Animal_194_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000filtered",  # Control
])



# Ensure `unique_names_selected` matches `moseq_df['name']` format
unique_names_selected_fixed = [
    name.replace(' ', '_') for name in unique_names_selected  # Ensure underscores match
]

# Debugging loop to check if filtering works
max_plots = 4
plots = 0
for animal_name in unique_names_selected_fixed:
    if plots >= max_plots:
        break
    plots += 1
    # Filter the dataframe
    df_animal = moseq_df[moseq_df['name'] == animal_name]

    # Derive the "base name" to look up in `freezing_data`
    base_name = parse_moseq_name(animal_name)
    if base_name not in freezing_data:
        print(f"No freezing CSV for {base_name} - skipping shading.")
        df_freeze = None
    else:
        df_freeze = freezing_data[base_name]

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 4))

    # Scatter all syllables vs. frame_index
    ax.scatter(df_animal['frame_index'], df_animal['syllable'], 
               s=1, c='black', label='All syllables')

    # Highlight specific syllables (0 and 28) with different colors
    special_syllables = [0, 28]
    colors = ['blue', 'red']
    markers = ['x', '^']
    for syllable, color, marker in zip(special_syllables, colors, markers):
        df_syll = df_animal[df_animal['syllable'] == syllable]
        ax.scatter(df_syll['frame_index'], df_syll['syllable'],
                   s=50, c=color, marker=marker, label=f'Syllable {syllable}')

    # -------------------------------------------------------
    # Shade freezing frames if we have a matching CSV
    # -------------------------------------------------------
    # df_freeze expected to have columns: 'Unnamed: 0', 'Freezing_Jen_0-125_threshold'
    if df_freeze is not None and 'Freezing_Jen_0-125_threshold' in df_freeze.columns:
        # Get all frame indices where "Freezing_Jen_0-125_threshold" == 1
        freeze_indices = df_freeze.loc[
            df_freeze['Freezing_Jen_0-125_threshold'] == 1, 'Unnamed: 0'
        ].to_numpy()

        if len(freeze_indices) > 0:
            # Identify consecutive runs (start->end) of freezing
            runs = []
            start = freeze_indices[0]
            end = start

            for fidx in freeze_indices[1:]:
                if fidx == end + 1:
                    # continue run
                    end = fidx
                else:
                    # close the old run, start a new run
                    runs.append((start, end))
                    start = fidx
                    end = fidx
            # Add the last run
            runs.append((start, end))

            # Shade each run with axvspan (in blue, low alpha)
            for (run_start, run_end) in runs:
                ax.axvspan(run_start, run_end + 1, color='blue', alpha=0.1, label='_nolegend_')

    # Final labeling
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Syllable')
    ax.set_title(f'Syllable Over Time: {animal_name}')
    ax.legend()
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





