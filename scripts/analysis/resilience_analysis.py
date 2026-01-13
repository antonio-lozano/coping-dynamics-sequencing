#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:49:14 2025

@author: lozano
"""

#%%
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from scipy.special import erf
from src.config import RESULTS_CLUSTERS_PKL, INDEX_CSV

def load_pickle(filename):
    """Load a pickle file and return its content."""
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_group_assignments(csv_file):
    """Load group assignments from a CSV assumed to have 'name' and 'group' columns."""
    index_df = pd.read_csv(csv_file)
    return dict(zip(index_df["name"], index_df["group"]))

def compute_transition_matrix(seq):
    """
    Remove consecutive duplicates from a sequence and compute its transition matrix.
    Returns a pandas DataFrame with rows = from-states and columns = to-states.
    """
    if len(seq) == 0:
        return pd.DataFrame()
    filtered = [seq[0]]
    for s in seq[1:]:
        if s != filtered[-1]:
            filtered.append(s)
    pairs = list(zip(filtered[:-1], filtered[1:]))
    states = sorted(set(filtered))
    TM = pd.DataFrame(0, index=states, columns=states, dtype=float)
    for a, b in pairs:
        TM.loc[a, b] += 1
    return TM

def compute_stabilized_transition_matrices(trans_mats, group_assignments, control_value):
    """
    Given:
      - trans_mats: dict mapping recording names to transition matrices (DataFrames)
      - group_assignments: dict mapping recording names to group labels
      - control_value: string label for the control group (e.g. "Control")
    Computes the average control transition matrix and subtracts it from each recording's matrix.
    Returns a dictionary mapping recording names to their stabilized transition matrices.
    """
    control_keys = [rec for rec, grp in group_assignments.items() if grp.lower() == control_value.lower()]
    if not control_keys:
        raise ValueError("No recordings found for the control group.")
    avg_control = None
    for rec in control_keys:
        if rec in trans_mats:
            avg_control = trans_mats[rec].copy() if avg_control is None else avg_control + trans_mats[rec]
    avg_control /= len(control_keys)

    stabilized = {}
    for rec, tm in trans_mats.items():
        tm_aligned = tm.reindex(index=avg_control.index, columns=avg_control.columns, fill_value=0)
        stabilized[rec] = tm_aligned - avg_control
    return stabilized

def relabel_matrix(tm, mapping):
    """Relabel a transition matrix's rows and columns using the provided mapping."""
    tm = tm.copy()
    tm.rename(index=mapping, columns=mapping, inplace=True)
    return tm

def compute_group_mean(mats):
    """Compute the element‐wise mean of a list of matrices."""
    return np.mean(np.stack(mats, axis=0), axis=0)

def compute_group_median(mats):
    """Compute the element‐wise median of a list of matrices."""
    return np.median(np.stack(mats, axis=0), axis=0)

def compute_group_mean(mats):
    """Compute the element‐wise median of a list of matrices."""
    return np.mean(np.stack(mats, axis=0), axis=0)

def manhattan_distance(mat1, mat2):
    """Compute the Manhattan distance (L1 norm) between two matrices."""
    arr1 = mat1.values if hasattr(mat1, 'values') else np.array(mat1)
    arr2 = mat2.values if hasattr(mat2, 'values') else np.array(mat2)
    return np.sum(np.abs(arr1 - arr2))

def bootstrap_intergroup_distance(transition_mats, group_assignments, group1, group2, n_bootstraps=1000, seed=123):
    """
    Given:
      - transition_mats: dict mapping recording names to a NumPy array (transition matrix)
      - group_assignments: dict mapping recording names to group labels
      - group1, group2: strings naming the two groups to compare.
    This function computes:
      - true_distance: Manhattan distance between group1 and group2 mean transition matrices.
      - A bootstrap null distribution by randomly shuffling the group labels.
      - The percentile (proportion of bootstrap distances less than true_distance),
        sigma (z-score) and a one-tailed p-value.
    Returns a dictionary with these statistics and the bootstrap distances.
    """
    recs = [rec for rec in transition_mats if group_assignments.get(rec, None) in (group1, group2)]
    recs_g1 = [rec for rec in recs if group_assignments[rec] == group1]
    recs_g2 = [rec for rec in recs if group_assignments[rec] == group2]
    if not recs_g1 or not recs_g2:
        raise ValueError("One of the groups has no recordings.")
    mean1 = compute_group_mean([transition_mats[rec] for rec in recs_g1])
    mean2 = compute_group_mean([transition_mats[rec] for rec in recs_g2])
    true_distance = manhattan_distance(mean1, mean2)

    recs_array = np.array(recs)
    original_labels = np.array([group_assignments[rec] for rec in recs])
    bootstrap_dists = []
    random.seed(seed)
    np.random.seed(seed)
    for _ in range(n_bootstraps):
        permuted = np.random.permutation(original_labels)
        boot_g1 = recs_array[permuted == group1]
        boot_g2 = recs_array[permuted == group2]
        if boot_g1.size == 0 or boot_g2.size == 0:
            continue
        mean_boot1 = compute_group_mean([transition_mats[rec] for rec in boot_g1])
        mean_boot2 = compute_group_mean([transition_mats[rec] for rec in boot_g2])
        bootstrap_dists.append(manhattan_distance(mean_boot1, mean_boot2))
    bootstrap_dists = np.array(bootstrap_dists)
    boot_mean = np.mean(bootstrap_dists)
    boot_std = np.std(bootstrap_dists)
    percentile = (np.sum(bootstrap_dists < true_distance) / (len(bootstrap_dists) + 1)) * 100
    sigma = (true_distance - boot_mean) / boot_std if boot_std > 0 else np.nan
    p_value = 0.5*(1 - erf(sigma/np.sqrt(2))) if not np.isnan(sigma) else np.nan

    stats = {
        'true_distance': true_distance,
        'bootstrap_mean': boot_mean,
        'bootstrap_std': boot_std,
        'percentile': percentile,
        'sigma': sigma,
        'p_value': p_value,
        'bootstrap_distances': bootstrap_dists
    }
    return stats

def compute_bfl_scores(transition_mats, group_assignments, group1, group2):
    """
    For each recording, compute its BFL score as follows:
      - Compute the elementwise median transition matrix for group1 and group2.
      - For each recording, compute d_A = Manhattan distance between its matrix and median_A,
        and d_B = Manhattan distance between its matrix and median_B.
      - BFL score = log(d_A / d_B)
    Returns a dictionary mapping recording names to BFL scores.
    """
    recs_g1 = [rec for rec in transition_mats if group_assignments.get(rec, None) == group1]
    recs_g2 = [rec for rec in transition_mats if group_assignments.get(rec, None) == group2]
    if not recs_g1 or not recs_g2:
        raise ValueError("One of the groups has no recordings.")
    median1 = compute_group_median([transition_mats[rec] for rec in recs_g1])
    median2 = compute_group_median([transition_mats[rec] for rec in recs_g2])

    bfl_scores = {}
    for rec, mat in transition_mats.items():
        if group_assignments.get(rec, None) not in (group1, group2):
            continue
        dA = manhattan_distance(mat, median1)
        dB = manhattan_distance(mat, median2)
        eps = 1e-10
        bfl = np.log((dA + eps) / (dB + eps))
        bfl_scores[rec] = bfl
    return bfl_scores

#%%
# === File Paths ===
results_file = RESULTS_CLUSTERS_PKL
index_csv = INDEX_CSV

# === Load Data ===
results_dict = load_pickle(results_file)
print("Loaded results from", results_file)
group_assignments = load_group_assignments(index_csv)
print("Loaded group assignments:")
print(group_assignments)

# === Define Behavior Mappings ===
new_numeric_mapping = {
    "Freezing": 1,
    "Turn": 4,
    "Locomotion": 5,
    "Sniffing": 2,
    "Climbing": 6,
    "Jump": 7,
    "Grooming": 3
}
behavior_mapping = {v: k for k, v in new_numeric_mapping.items()}
ordered_behaviors = sorted(new_numeric_mapping.items(), key=lambda x: x[1])
ordered_behaviors = [item for item in ordered_behaviors if 1 <= item[1] <= 7]
behavior_names = [name for name, code in ordered_behaviors]
behavior_codes = [code for name, code in ordered_behaviors]
print("Behavior names (ordered):", behavior_names)
print("Behavior codes (ordered):", behavior_codes)

# === Count Frames per Behavior (Bar Plot) ===
animal_names = []
counts_list = []
for animal, data in results_dict.items():
    if "syllable" not in data:
        continue
    animal_names.append(animal)
    syl = np.array(data["syllable"])
    unique_codes, unique_counts = np.unique(syl, return_counts=True)
    count_dict = dict(zip(unique_codes, unique_counts))
    counts = [count_dict.get(code, 0) for code in behavior_codes]
    counts_list.append(counts)

df_counts = pd.DataFrame(np.array(counts_list), index=animal_names, columns=behavior_names)
print("Summary of frames per behavior for each animal:")
print(df_counts)

mean_values = df_counts.mean(axis=0)
std_values = df_counts.std(axis=0)
plt.figure(figsize=(10, 6))
x = np.arange(len(mean_values))
plt.bar(x, mean_values, width=0.6, color="skyblue", edgecolor="black", yerr=std_values, capsize=5)
plt.xticks(x, mean_values.index, rotation=45, ha="right")
plt.ylabel("Number of Frames")
plt.xlabel("Behavior")
plt.title("Mean ± STD of Frames per Behavior")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

#%%
# === Compute Transition Matrices ===
selected_keys = [k for k in results_dict.keys() if k in group_assignments]
if not selected_keys:
    raise ValueError("No recordings with group information found!")

transition_matrices = {}
for rec in selected_keys:
    rec_data = results_dict[rec]
    if "syllable" not in rec_data:
        print(f"Recording {rec} does not contain a 'syllable' array. Skipping.")
        continue
    seq = rec_data["syllable"]
    TM = compute_transition_matrix(seq)
    TM = TM.reindex(index=behavior_codes, columns=behavior_codes, fill_value=0)
    transition_matrices[rec] = TM
if not transition_matrices:
    raise ValueError("No transition matrices computed. Check that recordings have a 'syllable' key.")

# === Compute Stabilized Transition Matrices ===
control_value = "Control"
stabilized_matrices = compute_stabilized_transition_matrices(transition_matrices, group_assignments, control_value)

# === Plot Example Stabilized Matrices for One Recording per Group ===
rec_control = None
rec_other = None
for rec in selected_keys:
    grp = group_assignments[rec]
    if grp.lower() == control_value.lower() and rec in stabilized_matrices and rec_control is None:
        rec_control = rec
    elif grp.lower() != control_value.lower() and rec in stabilized_matrices and rec_other is None:
        rec_other = rec

if rec_control is None or rec_other is None:
    print("Could not find one recording for each group.")
else:
    tm_other = relabel_matrix(stabilized_matrices[rec_other], behavior_mapping)
    tm_control = relabel_matrix(stabilized_matrices[rec_control], behavior_mapping)
    common_min = min(tm_other.min().min(), tm_control.min().min())
    common_max = max(tm_other.max().max(), tm_control.max().max())
    print(f"Stabilized transition matrix for {rec_other} (group: {group_assignments[rec_other]}):")
    print(tm_other)
    print(f"\nStabilized transition matrix for {rec_control} (group: {group_assignments[rec_control]}):")
    print(tm_control)

    plt.figure(figsize=(8, 6))
    sns.heatmap(tm_other, annot=True, cmap="coolwarm", vmin=common_min, vmax=common_max)
    plt.title(f"Stabilized Transition Matrix for {rec_other}\nGroup: {group_assignments[rec_other]}")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(tm_control, annot=True, cmap="coolwarm", vmin=common_min, vmax=common_max)
    plt.title(f"Stabilized Transition Matrix for {rec_control}\nGroup: {group_assignments[rec_control]}")
    plt.show()

#%%
# === Mean Transition Matrices per Group (Raw) ===
group_sums = {}
group_counts = {}
for rec, tm in transition_matrices.items():
    grp = group_assignments[rec]
    if grp not in group_sums:
        group_sums[grp] = tm.copy()
        group_counts[grp] = 1
    else:
        group_sums[grp] += tm
        group_counts[grp] += 1
group_means = {}
for grp in group_sums:
    mean_tm = group_sums[grp] / group_counts[grp]
    mean_tm = mean_tm.rename(index=behavior_mapping, columns=behavior_mapping)
    group_means[grp] = mean_tm
all_values = np.concatenate([tm.values.flatten() for tm in group_means.values()])
vmin_mean = np.nanmin(all_values)
vmax_mean = np.nanmax(all_values)

for grp, mean_tm in group_means.items():
    print(f"Mean transition matrix for group '{grp}':")
    print(mean_tm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(mean_tm, annot=True, cmap="coolwarm", vmin=vmin_mean, vmax=vmax_mean)
    plt.title(f"Mean Transition Matrix for Group '{grp}'")
    plt.xlabel("To Behavior")
    plt.ylabel("From Behavior")
    plt.show()

# === Mean Stabilized Transition Matrices per Group ===
group_sums_stab = {}
group_counts_stab = {}
for rec, tm in stabilized_matrices.items():
    grp = group_assignments[rec]
    if grp not in group_sums_stab:
        group_sums_stab[grp] = tm.copy()
        group_counts_stab[grp] = 1
    else:
        group_sums_stab[grp] += tm
        group_counts_stab[grp] += 1
group_means_stab = {}
for grp in group_sums_stab:
    mean_stab = group_sums_stab[grp] / group_counts_stab[grp]
    mean_stab = mean_stab.rename(index=behavior_mapping, columns=behavior_mapping)
    group_means_stab[grp] = mean_stab
all_values_stab = np.concatenate([tm.values.flatten() for tm in group_means_stab.values()])
abs_max = np.nanmax(np.abs(all_values_stab))
vmin_stab = -abs_max
vmax_stab = abs_max

for grp, mean_stab in group_means_stab.items():
    print(f"Mean stabilized transition matrix for group '{grp}':")
    print(mean_stab)
    plt.figure(figsize=(6, 5))
    sns.heatmap(mean_stab, annot=True, cmap="seismic_r", vmin=vmin_stab, vmax=vmax_stab)
    plt.title(f"Mean Stabilized Transition Matrix for Group '{grp}'")
    plt.xlabel("To Behavior")
    plt.ylabel("From Behavior")
    plt.show()

# === Behavioral Flow Analysis ===
stats = bootstrap_intergroup_distance(stabilized_matrices, group_assignments, 'Control', 'ELS', n_bootstraps=1000)
print("Intergroup Manhattan distance statistics:")
for k, v in stats.items():
    if k != 'bootstrap_distances':
        print(f"{k}: {v}")

plt.figure(figsize=(8, 4))
plt.hist(stats['bootstrap_distances'], bins=30, color='gray', edgecolor='black', alpha=0.7)
plt.axvline(stats['true_distance'], color='red', linewidth=2, label='True Distance')
plt.xlabel('Manhattan Distance')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution of Manhattan Distances')
plt.legend()
plt.tight_layout()
plt.show()

#%%
# Compute BFL scores for each recording.
bfl_scores = compute_bfl_scores(stabilized_matrices, group_assignments, 'Control', 'ELS')
print("\nBehavioral Flow Likeness (BFL) scores:")
for rec, score in bfl_scores.items():
    print(f"{rec}: {score:.3f}")

groups = {'Control': [], 'ELS': []}
for rec, score in bfl_scores.items():
    grp = group_assignments[rec]
    groups[grp].append(score)
plt.figure(figsize=(6, 4))
for grp, scores in groups.items():
    plt.hist(scores, bins=10, alpha=0.6, label=grp)
plt.xlabel('BFL score (log(dA/dB))')
plt.ylabel('Count')
plt.title('Distribution of BFL scores by group')
plt.legend()
plt.tight_layout()
plt.show()

#%%
import scipy.stats as st

# Create a DataFrame from the bfl_scores grouped by group.
all_scores = []
for grp, scores in groups.items():
    for s in scores:
        all_scores.append({'group': grp, 'score': s})
df_bfl = pd.DataFrame(all_scores)

# Compute mean and SEM per group.
group_names = df_bfl['group'].unique()
means = df_bfl.groupby('group')['score'].mean().loc[group_names]
sems = df_bfl.groupby('group')['score'].apply(st.sem).loc[group_names]

# Perform one-way ANOVA
group_scores = [group_data['score'].values for name, group_data in df_bfl.groupby('group')]
f_statistic, p_value = st.f_oneway(*group_scores)
print(f"ANOVA: F = {f_statistic:.3f}, p = {p_value:.3f}")

# Plot barplot using matplotlib.
plt.figure(figsize=(6,4))
bars = plt.bar(group_names, means, yerr=sems, capsize=10, color=['skyblue', 'salmon'])
plt.ylabel('Mean BFL Score')
plt.title('Behavioral Flow Likeness (BFL) per Group')

# If statistically significant (p < 0.05), add a star annotation.
if p_value < 0.05:
    # Get the center x positions for each bar.
    x_coords = [bar.get_x() + bar.get_width()/2 for bar in bars]
    # Determine the highest value among bars (mean + error) for placement.
    y_max = max(means + sems)
    y_star = y_max + 0.1 * y_max  # 10% above the max value
    # For two groups, place the star at the center between the two bars.
    x_center = np.mean(x_coords)
    plt.text(x_center, y_star, '*', ha='center', va='bottom', color='black', fontsize=20)

plt.tight_layout()
plt.show()


#%% B FLOW ANALYSIS PER TIME (FIRST 3 MIN, LATER EXPERIMENT SEPARATED)
# --- Set frame rate to 25 fps and compute the number of frames for 3 minutes ---
frame_rate = 25
frames_first3 = frame_rate * 180  # 180 seconds = 3 minutes → 25 * 180 = 4500 frames




#%% NUEVA MISTRAL
# --- Provided Lists ---
exp1_list = [
    ("Animal 2_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 5_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 11_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 11_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 15_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 15_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 23_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 25_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 26_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 32_2DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 32_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 34_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 34_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 40_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 40_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 41_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 41_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 41_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 43_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 43_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 47_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 49_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 49_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 50_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 53_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 54_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 54_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 54_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 55_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 55_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 56_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 58_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 62_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 62_2DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 62_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 62_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 62_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 63_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 68_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 69_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 69_2DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 69_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 69_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 73_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 73_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 74_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 75_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 80_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 88_2DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 88_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS")
]

exp3_list = [
    ("Animal_123_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_123_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_128_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_129_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_129_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_133_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_134_2DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_134_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_136_2DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_136_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_136_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_137_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_137_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_144_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_144_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_145_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_146_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_147_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_148_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_150_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_150_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_152_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_152_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_153_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_154_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_157_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_157_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_158_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_159_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_159_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_159_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_161_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_161_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control")
]


# --- Combine the Lists ---
records = []
for rec, grp in exp1_list:
    records.append({"recording": rec, "experiment": "1", "group": grp})
for rec, grp in exp3_list:
    records.append({"recording": rec, "experiment": "3", "group": grp})

df = pd.DataFrame(records)

# --- Adjust Recording Keys to Match bfl_scores ---
# Helper function: if the key is missing, try appending "filtered".
def adjust_key(rec, keys):
    if rec in keys:
        return rec
    elif not rec.endswith("filtered") and (rec + "filtered") in keys:
        return rec + "filtered"
    else:
        return rec

df["recording"] = df["recording"].apply(lambda rec: adjust_key(rec, bfl_scores.keys()))

# Retain only recordings that are present in bfl_scores.
df = df[df["recording"].isin(bfl_scores.keys())].copy()
df["score"] = df["recording"].map(bfl_scores)

# --- Combine recordings from Experiments 1 and 3 ---
combined_recs = df[df["experiment"].isin(["1", "3"])]["recording"].unique()

# Dictionaries to hold transition matrices and group assignments for each segment
trans_mats_first3 = {}
group_assignments_first3 = {}
trans_mats_rest = {}
group_assignments_rest = {}

# Loop over all combined recordings
for rec in combined_recs:
    # Ensure the recording exists and contains a "syllable" array
    if rec not in results_dict or "syllable" not in results_dict[rec]:
        continue
    seq = results_dict[rec]["syllable"]
    
    # --- First 3 Minutes ---
    if len(seq) >= frames_first3:
        first_seq = seq[:frames_first3]
        TM_first = compute_transition_matrix(first_seq)
        TM_first = TM_first.reindex(index=behavior_codes, columns=behavior_codes, fill_value=0)
        trans_mats_first3[rec] = TM_first
        group_assignments_first3[rec] = group_assignments[rec]
    
    # --- Rest of the Recording ---
    if len(seq) > frames_first3:
        rest_seq = seq[frames_first3:]
        if len(rest_seq) > 0:
            TM_rest = compute_transition_matrix(rest_seq)
            TM_rest = TM_rest.reindex(index=behavior_codes, columns=behavior_codes, fill_value=0)
            trans_mats_rest[rec] = TM_rest
            group_assignments_rest[rec] = group_assignments[rec]

# --- Analyze each segment (First 3 Minutes and Rest) for the combined experiments ---
for segment_name, trans_mats_seg, group_assignments_seg in [("First 3 Minutes", trans_mats_first3, group_assignments_first3),
                                                            ("Experiment", trans_mats_rest, group_assignments_rest)]:
    if len(trans_mats_seg) == 0:
        print(f"No data available for segment: {segment_name}")
        continue
    
    # Compute stabilized transition matrices for the current segment
    stabilized_seg = compute_stabilized_transition_matrices(trans_mats_seg, group_assignments_seg, control_value)
    
    # Compute BF (BFL) scores for each recording in this segment
    bfl_scores_seg = compute_bfl_scores(stabilized_seg, group_assignments_seg, 'Control', 'ELS')
    df_bfl_seg = pd.DataFrame([{"recording": rec, "score": score, "group": group_assignments_seg[rec]} 
                               for rec, score in bfl_scores_seg.items()])
    
    # --- Statistical Analysis: t-test comparing Control and ELS ---
    control_scores = df_bfl_seg[df_bfl_seg["group"] == "Control"]["score"].values
    els_scores = df_bfl_seg[df_bfl_seg["group"] == "ELS"]["score"].values
    mean_control = np.mean(control_scores) if control_scores.size > 0 else np.nan
    sem_control = st.sem(control_scores) if control_scores.size > 1 else np.nan
    mean_els = np.mean(els_scores) if els_scores.size > 0 else np.nan
    sem_els = st.sem(els_scores) if els_scores.size > 1 else np.nan
    if control_scores.size > 1 and els_scores.size > 1:
        t_stat, p_val = st.ttest_ind(control_scores, els_scores, equal_var=False)
    else:
        t_stat, p_val = np.nan, np.nan
        
    print(f"\nCombined (Exp 1 & 3) - {segment_name} statistics:")
    print(f"  Control (n={len(control_scores)}): mean = {mean_control:.3f}, SEM = {sem_control:.3f}")
    print(f"  ELS     (n={len(els_scores)}): mean = {mean_els:.3f}, SEM = {sem_els:.3f}")
    print(f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.3f}")
    
    # --- Bar Plot: Group Means with Individual Data Points and Significance Annotation ---
    groups_order = ["Control", "ELS"]
    means = [mean_control, mean_els]
    sems = [sem_control, sem_els]
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(groups_order, means, yerr=sems, capsize=10,
                   color=["skyblue", "salmon"], edgecolor="black")
    plt.ylabel("Mean BFL Score")
    plt.title(f"Combined BFL Scores by Group ({segment_name})")
    
    # Overlay individual data points (with slight horizontal jitter)
    for i, grp in enumerate(groups_order):
        group_points = df_bfl_seg[df_bfl_seg["group"] == grp]["score"].values
        x_bar = bars[i].get_x() + bars[i].get_width() / 2
        jitter = np.random.uniform(-0.05, 0.05, size=len(group_points))
        plt.scatter(np.full(len(group_points), x_bar) + jitter, group_points,
                    color="black", zorder=10, s=50, alpha=0.8)
    
    # Annotate p-value above the bars
    y_max = max([m + s for m, s in zip(means, sems)])
    y_annotate = y_max + 0.1 * y_max
    x_center = np.mean([bar.get_x() + bar.get_width() / 2 for bar in bars])
    plt.text(x_center, y_annotate, f"p = {p_val:.3f}", ha="center", va="bottom", fontsize=12)
    
    # If the difference is significant, add a significance star with connecting lines
    if p_val < 0.05:
        x1 = bars[0].get_x() + bars[0].get_width() / 2
        x2 = bars[1].get_x() + bars[1].get_width() / 2
        line_y = y_max + 0.05 * y_max
        plt.plot([x1, x1, x2, x2],
                 [line_y, line_y + 0.02 * y_max, line_y + 0.02 * y_max, line_y],
                 lw=1.5, c="black")
        plt.text((x1 + x2) / 2, line_y + 0.02 * y_max + 0.01 * y_max,
                 "*", ha="center", va="bottom", color="black", fontsize=20)
    
    plt.tight_layout()
    plt.show()
    
    # --- Bootstrap Analysis ---
    stats_seg = bootstrap_intergroup_distance(stabilized_seg, group_assignments_seg, 'Control', 'ELS', n_bootstraps=1000)
    print(f"\nCombined (Exp 1 & 3) - {segment_name} Bootstrap Analysis:")
    for k, v in stats_seg.items():
        if k != 'bootstrap_distances':
            print(f"{k}: {v}")
    plt.figure(figsize=(8, 4))
    plt.hist(stats_seg["bootstrap_distances"], bins=30, color="gray", edgecolor="black", alpha=0.7)
    plt.axvline(stats_seg["true_distance"], color="red", linewidth=2, label="True Distance")
    plt.xlabel("Manhattan Distance")
    plt.ylabel("Frequency")
    plt.title(f"Combined Bootstrap Distribution ({segment_name})")
    plt.legend()
    plt.tight_layout()
    plt.show()





#%% linear mixed model to have the experiment differences in account
import pandas as pd
import re
import statsmodels.formula.api as smf

#%% NUEVA MISTRAL
# --- Provided Lists ---
exp1_list = [
    ("Animal 11_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 11_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 15_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 15_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 23_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 25_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 26_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 2_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 32_2DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 32_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal 34_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 34_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal 5_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_40_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_40_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_41_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_41_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_41_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_43_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_43_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_47_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_49_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_49_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_50_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_53_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_54_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_54_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_54_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_55_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_55_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_56_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_58_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_62_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_62_2DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_62_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_62_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_62_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_63_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_68_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_69_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_69_2DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_69_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_69_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_73_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_73_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_74_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_75_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_80_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_88_2DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_88_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS")
]

exp3_list = [
    ("Animal_123_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_123_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_128_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_129_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_129_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_133_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_134_2DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_134_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_136_2DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_136_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_136_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_137_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_144_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_144_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_145_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_146_6DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_147_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_148_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_150_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_150_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_152_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_152_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_153_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_154_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_157_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_157_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_158_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_159_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_159_4DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_159_5DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "ELS"),
    ("Animal_161_1DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control"),
    ("Animal_161_3DLC_resnet50_Freezing_07-2020Sep29shuffle1_100000", "Control")
]


# --- Combine the Lists ---
records = []
for rec, grp in exp1_list:
    records.append({"recording": rec, "experiment": "1", "group": grp})
for rec, grp in exp3_list:
    records.append({"recording": rec, "experiment": "3", "group": grp})

df = pd.DataFrame(records)

# --- Adjust Recording Keys to Match bfl_scores ---
# Helper function: if the key is missing, try appending "filtered".
def adjust_key(rec, keys):
    if rec in keys:
        return rec
    elif not rec.endswith("filtered") and (rec + "filtered") in keys:
        return rec + "filtered"
    else:
        return rec

df["recording"] = df["recording"].apply(lambda rec: adjust_key(rec, bfl_scores.keys()))

# --- Adjust Recording Keys to Match transition_matrices ---
# (Assuming transition_matrices contains all recordings you want to include.)
df["recording"] = df["recording"].apply(lambda rec: adjust_key(rec, transition_matrices.keys()))


# Retain only recordings that are present in bfl_scores.
df = df[df["recording"].isin(bfl_scores.keys())].copy()
df["score"] = df["recording"].map(bfl_scores)


# --- Use the provided lists to define the experiment keys (after adjusting keys) ---
exp1_keys = set(adjust_key(rec, transition_matrices.keys()) for rec, _ in exp1_list) & set(bfl_scores.keys())
exp3_keys = set(adjust_key(rec, transition_matrices.keys()) for rec, _ in exp3_list) & set(bfl_scores.keys())

# --- Filter the transition matrices and group assignments for each experiment ---
transition_matrices_exp1 = {rec: tm for rec, tm in transition_matrices.items() if rec in exp1_keys}
group_assignments_exp1   = {rec: grp for rec, grp in group_assignments.items() if rec in exp1_keys}

transition_matrices_exp3 = {rec: tm for rec, tm in transition_matrices.items() if rec in exp3_keys}
group_assignments_exp3   = {rec: grp for rec, grp in group_assignments.items() if rec in exp3_keys}

# --- Compute stabilized transition matrices and BF scores ---
stabilized_exp1 = compute_stabilized_transition_matrices(transition_matrices_exp1, group_assignments_exp1, control_value)
stabilized_exp3 = compute_stabilized_transition_matrices(transition_matrices_exp3, group_assignments_exp3, control_value)

bfl_scores_exp1 = compute_bfl_scores(stabilized_exp1, group_assignments_exp1, 'Control', 'ELS')
bfl_scores_exp3 = compute_bfl_scores(stabilized_exp3, group_assignments_exp3, 'Control', 'ELS')

# --- For each experiment, build a DataFrame, print the scores, and assert expected counts ---
for exp_label, bfl_dict, grp_assign in zip(["1", "3"],
                                             [bfl_scores_exp1, bfl_scores_exp3],
                                             [group_assignments_exp1, group_assignments_exp3]):
    df_bfl = pd.DataFrame([{"recording": rec, "score": score, "group": grp_assign[rec]}
                           for rec, score in bfl_dict.items()])
    
    control_scores = df_bfl[df_bfl["group"] == "Control"]["score"].values
    els_scores     = df_bfl[df_bfl["group"] == "ELS"]["score"].values
    

    print(f"\nBF scores for Experiment {exp_label}:")
    print("Control scores:", len(control_scores))
    print("ELS scores:", len(els_scores))


    # Compute descriptive statistics for each group
    mean_control = np.mean(control_scores) if control_scores.size > 0 else np.nan
    sem_control  = st.sem(control_scores)  if control_scores.size > 1 else np.nan
    mean_els     = np.mean(els_scores)     if els_scores.size > 0 else np.nan
    sem_els      = st.sem(els_scores)      if els_scores.size > 1 else np.nan
    
    # Perform an independent t-test (unequal variances assumed) if both groups have enough samples
    if control_scores.size > 1 and els_scores.size > 1:
        t_stat, p_val = st.ttest_ind(control_scores, els_scores, equal_var=False)
    else:
        t_stat, p_val = np.nan, np.nan
    
    print(f"\nExperiment {exp_label} statistics:")
    print(f"  Control (n={len(control_scores)}): mean = {mean_control:.3f}, SEM = {sem_control:.3f}")
    print(f"  ELS     (n={len(els_scores)}): mean = {mean_els:.3f}, SEM = {sem_els:.3f}")
    print(f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.3f}")
    
    # Create a bar plot of BF scores per group with individual data points
    groups_order = ["Control", "ELS"]
    means = [mean_control, mean_els]
    sems  = [sem_control, sem_els]
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(groups_order, means, yerr=sems, capsize=10,
                   color=["skyblue", "salmon"], edgecolor="black")
    plt.ylabel("Mean BFL Score")
    plt.title(f"BFL Scores by Group — Experiment {exp_label}")
    
    # Overlay individual BF score data points (with slight horizontal jitter)
    for i, grp in enumerate(groups_order):
        group_points = df_bfl[df_bfl["group"] == grp]["score"].values
        # Calculate the center x position of the bar
        x_bar = bars[i].get_x() + bars[i].get_width() / 2
        # Create small random horizontal jitter for each data point
        jitter = np.random.uniform(-0.05, 0.05, size=len(group_points))
        plt.scatter(np.full(len(group_points), x_bar) + jitter, group_points,
                    color='black', zorder=10, s=50, alpha=0.8)
    
    # Determine y-position for annotations (10% above the highest mean+SEM)
    y_max = max([m + s for m, s in zip(means, sems)])
    y_annotate = y_max + 0.1 * y_max
    x_center = np.mean([bar.get_x() + bar.get_width() / 2 for bar in bars])
    
    # Annotate the plot with the p-value
    plt.text(x_center, y_annotate, f"p = {p_val:.3f}", ha="center", va="bottom", fontsize=12)
    
    # If significant, add a significance star with a connecting line between bars
    if p_val < 0.05:
        # Get x positions of both bars
        x1 = bars[0].get_x() + bars[0].get_width() / 2
        x2 = bars[1].get_x() + bars[1].get_width() / 2
        # y-position for the line (a little above the highest error bar)
        line_y = y_max + 0.05 * y_max
        # Draw vertical lines at each bar and a horizontal line connecting them
        plt.plot([x1, x1, x2, x2],
                 [line_y, line_y + 0.02 * y_max, line_y + 0.02 * y_max, line_y],
                 lw=1.5, c='black')
        # Place a star at the center of the horizontal line
        plt.text((x1 + x2) / 2, line_y + 0.02 * y_max + 0.01 * y_max,
                 '*', ha='center', va='bottom', color='black', fontsize=20)
    
    plt.tight_layout()
    plt.show()


#%% Bootstrap distribution analysis per dexperiment

# === Bootstrap Distribution Analysis for Experiment 1 ===
stats_exp1 = bootstrap_intergroup_distance(stabilized_exp1, group_assignments_exp1, 'Control', 'ELS', n_bootstraps=1000)
print("Experiment 1: Intergroup Manhattan distance statistics:")
for k, v in stats_exp1.items():
    if k != 'bootstrap_distances':
        print(f"{k}: {v}")

plt.figure(figsize=(8, 4))
plt.hist(stats_exp1['bootstrap_distances'], bins=30, color='gray', edgecolor='black', alpha=0.7)
plt.axvline(stats_exp1['true_distance'], color='red', linewidth=2, label='True Distance')
plt.xlabel('Manhattan Distance')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution of Manhattan Distances — Experiment 1')
plt.legend()
plt.tight_layout()
plt.show()


# === Bootstrap Distribution Analysis for Experiment 3 ===
stats_exp3 = bootstrap_intergroup_distance(stabilized_exp3, group_assignments_exp3, 'Control', 'ELS', n_bootstraps=1000)
print("\nExperiment 3: Intergroup Manhattan distance statistics:")
for k, v in stats_exp3.items():
    if k != 'bootstrap_distances':
        print(f"{k}: {v}")

plt.figure(figsize=(8, 4))
plt.hist(stats_exp3['bootstrap_distances'], bins=30, color='gray', edgecolor='black', alpha=0.7)
plt.axvline(stats_exp3['true_distance'], color='red', linewidth=2, label='True Distance')
plt.xlabel('Manhattan Distance')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution of Manhattan Distances — Experiment 3')
plt.legend()
plt.tight_layout()
plt.show()

'''
Experiment 1: Intergroup Manhattan distance statistics:
true_distance: 14.143939393939391
bootstrap_mean: 18.097931818181817
bootstrap_std: 5.9928616722072565
percentile: 28.57142857142857
sigma: -0.6597836960895701
p_value: 0.7453036762794137

Experiment 3: Intergroup Manhattan distance statistics:
true_distance: 26.323529411764703
bootstrap_mean: 19.11976470588235
bootstrap_std: 6.572007861542766
percentile: 85.01498501498502
sigma: 1.0961284371000863
p_value: 0.13651128575347216
'''


######################################################################################################
#%% Separating experiments as well as first 3 minutes (exploring) from the later minutes of experiment
######################################################################################################


# --- Set the frame rate if not already defined ---
frame_rate = 25


# Number of frames corresponding to the first 3 minutes
frames_first3 = frame_rate * 180  # 3 minutes = 180 seconds

# Loop over experiments and then over the two time segments
for exp_label in ["1", "3"]:
    # Get list of recordings for the current experiment from df
    recs_exp = df[df["experiment"] == exp_label]["recording"].unique()
    
    # Dictionaries to hold transition matrices and group assignments for each segment
    trans_mats_first3 = {}
    group_assignments_first3 = {}
    trans_mats_rest = {}
    group_assignments_rest = {}
    
    # Loop through recordings in this experiment
    for rec in recs_exp:
        if rec not in results_dict or "syllable" not in results_dict[rec]:
            continue
        seq = results_dict[rec]["syllable"]
        
        # --- First 3 Minutes ---
        if len(seq) >= frames_first3:
            first_seq = seq[:frames_first3]
            # Compute transition matrix for the first segment
            TM_first = compute_transition_matrix(first_seq)
            TM_first = TM_first.reindex(index=behavior_codes, columns=behavior_codes, fill_value=0)
            trans_mats_first3[rec] = TM_first
            group_assignments_first3[rec] = group_assignments[rec]
        
        # --- Rest of the Recording ---
        if len(seq) > frames_first3:
            rest_seq = seq[frames_first3:]
            if len(rest_seq) > 0:
                TM_rest = compute_transition_matrix(rest_seq)
                TM_rest = TM_rest.reindex(index=behavior_codes, columns=behavior_codes, fill_value=0)
                trans_mats_rest[rec] = TM_rest
                group_assignments_rest[rec] = group_assignments[rec]
    
    # Analyze each segment separately
    for segment_name, trans_mats_seg, group_assignments_seg in [("First 3 Minutes", trans_mats_first3, group_assignments_first3),
                                                                ("Experiment", trans_mats_rest, group_assignments_rest)]:
        if len(trans_mats_seg) == 0:
            print(f"No data for Experiment {exp_label} segment: {segment_name}")
            continue
        
        # --- Compute Stabilized Transition Matrices for the Segment ---
        stabilized_seg = compute_stabilized_transition_matrices(trans_mats_seg, group_assignments_seg, control_value)
        
        # --- Compute BF (BFL) Scores ---
        bfl_scores_seg = compute_bfl_scores(stabilized_seg, group_assignments_seg, 'Control', 'ELS')
        df_bfl_seg = pd.DataFrame([{"recording": rec, "score": score, "group": group_assignments_seg[rec]} 
                                   for rec, score in bfl_scores_seg.items()])
        
        # --- Statistical Analysis ---
        control_scores = df_bfl_seg[df_bfl_seg["group"] == "Control"]["score"].values
        els_scores = df_bfl_seg[df_bfl_seg["group"] == "ELS"]["score"].values
        mean_control = np.mean(control_scores) if control_scores.size > 0 else np.nan
        sem_control = st.sem(control_scores) if control_scores.size > 1 else np.nan
        mean_els = np.mean(els_scores) if els_scores.size > 0 else np.nan
        sem_els = st.sem(els_scores) if els_scores.size > 1 else np.nan
        if control_scores.size > 1 and els_scores.size > 1:
            t_stat, p_val = st.ttest_ind(control_scores, els_scores, equal_var=False)
        else:
            t_stat, p_val = np.nan, np.nan
        
        print(f"\nExperiment {exp_label} - {segment_name} statistics:")
        print(f"  Control (n={len(control_scores)}): mean = {mean_control:.3f}, SEM = {sem_control:.3f}")
        print(f"  ELS     (n={len(els_scores)}): mean = {mean_els:.3f}, SEM = {sem_els:.3f}")
        print(f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.3f}")
        
        # --- Bar Plot with Individual Data Points and Significance Star ---
        groups_order = ["Control", "ELS"]
        means = [mean_control, mean_els]
        sems = [sem_control, sem_els]
        
        plt.figure(figsize=(6, 4))
        bars = plt.bar(groups_order, means, yerr=sems, capsize=10,
                       color=["skyblue", "salmon"], edgecolor="black")
        plt.ylabel("Mean BFL Score")
        plt.title(f"BFL Scores by Group — Experiment {exp_label} ({segment_name})")
        
        # Overlay individual BF score points with jitter
        for i, grp in enumerate(groups_order):
            group_points = df_bfl_seg[df_bfl_seg["group"] == grp]["score"].values
            x_bar = bars[i].get_x() + bars[i].get_width() / 2
            jitter = np.random.uniform(-0.05, 0.05, size=len(group_points))
            plt.scatter(np.full(len(group_points), x_bar) + jitter, group_points,
                        color='black', zorder=10, s=50, alpha=0.8)
        
        y_max = max([m + s for m, s in zip(means, sems)])
        y_annotate = y_max + 0.1 * y_max
        x_center = np.mean([bar.get_x() + bar.get_width()/2 for bar in bars])
        plt.text(x_center, y_annotate, f"p = {p_val:.3f}", ha="center", va="bottom", fontsize=12)
        
        # Add significance star if p < 0.05
        if p_val < 0.05:
            x1 = bars[0].get_x() + bars[0].get_width()/2
            x2 = bars[1].get_x() + bars[1].get_width()/2
            line_y = y_max + 0.05 * y_max
            plt.plot([x1, x1, x2, x2],
                     [line_y, line_y + 0.02 * y_max, line_y + 0.02 * y_max, line_y],
                     lw=1.5, c='black')
            plt.text((x1 + x2)/2, line_y + 0.02*y_max + 0.01*y_max,
                     '*', ha='center', va='bottom', color='black', fontsize=20)
        
        plt.tight_layout()
        plt.show()
        
        # --- Bootstrap Analysis ---
        stats_seg = bootstrap_intergroup_distance(stabilized_seg, group_assignments_seg, 'Control', 'ELS', n_bootstraps=1000)
        print(f"\nExperiment {exp_label} - {segment_name} Bootstrap Analysis:")
        for k, v in stats_seg.items():
            if k != 'bootstrap_distances':
                print(f"{k}: {v}")
        plt.figure(figsize=(8, 4))
        plt.hist(stats_seg['bootstrap_distances'], bins=30, color='gray', edgecolor='black', alpha=0.7)
        plt.axvline(stats_seg['true_distance'], color='red', linewidth=2, label='True Distance')
        plt.xlabel('Manhattan Distance')
        plt.ylabel('Frequency')
        plt.title(f"Bootstrap Distribution — Experiment {exp_label} ({segment_name})")
        plt.legend()
        plt.tight_layout()
        plt.show()

'''
Overall Summary
Experiment 1:
During the first 3 minutes, there is only a modest (non-significant) trend in BF scores between groups, and the bootstrap analysis supports that the separation in transition matrices is not unusually large.
In the rest of the recording, the BF scores diverge significantly (with ELS showing higher values), although the bootstrap analysis of the transition matrices does not show a corresponding strong deviation from chance.
Experiment 3:
In the first 3 minutes, there is a very robust difference: Controls have clearly lower BF scores while ELS animals show much higher scores, and the t-test is highly significant. The bootstrap analysis trends in the same direction (observed distance is higher than most null values) but does not reach significance.
In the rest of the recording, the group differences become less pronounced (only a trend by t-test) and the bootstrap analysis again does not show a significant separation.
In short, for both experiments the early phase (first 3 minutes) appears more critical for revealing group differences in behavioral flow—especially in Experiment 3, where the early BF scores differ dramatically between Control and ELS animals. In Experiment 1, the divergence becomes significant only in the later period, but the bootstrap analyses suggest that differences in the raw transition matrices are not as extreme as the BF scores imply. These results highlight that the timing of the experiment matters: early exploration (first 3 minutes) might be driving stronger or more consistent behavioral differences between groups, at least in some experimental conditions.

'''


#%% DYNAMICAL BFL ANALYSIS. each BIN we calculate the BF scores

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

#%% DYNAMICAL BFL ANALYSIS (Line Plot with Mean & SEM Shaded)

# --- Set parameters ---
frame_rate = 25        # frames per second
bin_duration = 30      # seconds per bin
bin_size = frame_rate * bin_duration  # 25*30 = 750 frames per bin

# Define the colors for groups
color_control = "#f9c74f"  # Control
color_els = "#c37bac"      # ELS

# Process each experiment ("1" and "3") separately
for exp_label in ["1", "3"]:
    # Get recordings for this experiment (from df)
    recs_exp = df[df["experiment"] == exp_label]["recording"].unique()
    
    # Create a dictionary to store data for each time bin across recordings.
    # For each bin (indexed by 0, 1, 2, …) we will accumulate:
    #   - "trans_mats": dict of {recording: transition matrix for that bin}
    #   - "group_assignments": dict of {recording: group label}
    bins_dict = {}
    
    for rec in recs_exp:
        if rec not in results_dict or "syllable" not in results_dict[rec]:
            continue
        seq = results_dict[rec]["syllable"]
        num_bins = len(seq) // bin_size  # only complete bins
        for bin_idx in range(num_bins):
            start = bin_idx * bin_size
            end = (bin_idx + 1) * bin_size
            bin_seq = seq[start:end]
            # Compute the transition matrix for this bin and reindex to ensure consistency
            TM_bin = compute_transition_matrix(bin_seq)
            TM_bin = TM_bin.reindex(index=behavior_codes, columns=behavior_codes, fill_value=0)
            if bin_idx not in bins_dict:
                bins_dict[bin_idx] = {"trans_mats": {}, "group_assignments": {}}
            bins_dict[bin_idx]["trans_mats"][rec] = TM_bin
            bins_dict[bin_idx]["group_assignments"][rec] = group_assignments[rec]
    
    # For each bin, compute the stabilized matrices and BF scores.
    # We'll accumulate one row per recording per bin.
    bf_data = []
    for bin_idx, bin_data in bins_dict.items():
        # Compute stabilized transition matrices for this bin (control average computed from recordings in the bin)
        stabilized_bin = compute_stabilized_transition_matrices(bin_data["trans_mats"],
                                                                bin_data["group_assignments"],
                                                                control_value)
        # Compute BF (BFL) scores for the recordings in this bin
        bfl_scores_bin = compute_bfl_scores(stabilized_bin, bin_data["group_assignments"], 'Control', 'ELS')
        # Create a time bin label like "0-30", "30-60", etc.
        time_bin_label = f"{bin_idx*30}-{(bin_idx+1)*30}"
        for rec, score in bfl_scores_bin.items():
            bf_data.append({"Time_Bin": time_bin_label,
                            "Bin_Start": bin_idx*30,  # numeric start time
                            "Group": bin_data["group_assignments"][rec],
                            "Score": score})
    
    # Convert the accumulated data into a DataFrame
    df_bf = pd.DataFrame(bf_data)
    
    # Convert Time_Bin to a categorical ordered by Bin_Start
    df_bf = df_bf.sort_values("Bin_Start")
    unique_bins = df_bf["Time_Bin"].unique()
    
    # For each group and bin, compute mean and SEM
    stats_list = []
    for bin_label in unique_bins:
        for grp in ["Control", "ELS"]:
            sub = df_bf[(df_bf["Time_Bin"] == bin_label) & (df_bf["Group"] == grp)]
            mean_val = sub["Score"].mean()
            sem_val = sub["Score"].sem() if len(sub) > 1 else np.nan
            # Get the numeric bin center (average of start and end)
            start_time = int(bin_label.split("-")[0])
            center_time = start_time + bin_duration/2
            stats_list.append({"Time_Bin": bin_label, "Bin_Center": center_time,
                                "Group": grp, "Mean": mean_val, "SEM": sem_val})
    df_stats = pd.DataFrame(stats_list)
    
    # For each time bin, perform a t-test between Control and ELS
    ttest_results = {}
    for bin_label in unique_bins:
        sub = df_bf[df_bf["Time_Bin"] == bin_label]
        control_scores = sub[sub["Group"] == "Control"]["Score"].values
        els_scores = sub[sub["Group"] == "ELS"]["Score"].values
        if len(control_scores) > 1 and len(els_scores) > 1:
            t_stat, p_val = st.ttest_ind(control_scores, els_scores, equal_var=False)
        else:
            p_val = np.nan
        ttest_results[bin_label] = p_val

    # Plotting: Mean and SEM as a line with shaded error + individual dots.
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get unique bin centers (sorted)
    bin_centers = sorted(df_stats["Bin_Center"].unique())
    
    # For each group, get the mean and SEM per bin and plot the line with shading.
    for grp, col in zip(["Control", "ELS"], [color_control, color_els]):
        df_grp = df_stats[df_stats["Group"] == grp].sort_values("Bin_Center")
        x_vals = df_grp["Bin_Center"].values
        y_vals = df_grp["Mean"].values
        y_err = df_grp["SEM"].values
        # Plot the mean as a line with markers.
        ax.plot(x_vals, y_vals, marker="o", color=col, label=grp, linewidth=2)
        # Shade the area between (mean - sem) and (mean + sem)
        ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err, color=col, alpha=0.3)
    
    # Overlay individual data points with horizontal offsets to separate groups.
    for grp, col in zip(["Control", "ELS"], [color_control, color_els]):
        df_grp = df_bf[df_bf["Group"] == grp]
        # Set a fixed offset (in seconds) for each group: Control left, ELS right.
        offset = -4 if grp == "Control" else 4
        # Add a small random jitter for further separation.
        jitter = np.random.uniform(-2, 2, size=len(df_grp))
        # The x-coordinate is the bin center plus offset.
        ax.scatter(df_grp["Bin_Start"] + bin_duration/2 + offset + jitter, df_grp["Score"],
                   color=col, edgecolor = 'black', alpha=0.8, s=40, zorder=5)
    
    # Annotate p-values per time bin.
    for bin_label in unique_bins:
        sub = df_bf[df_bf["Time_Bin"] == bin_label]
        p_val = ttest_results[bin_label]
        # Compute the bin center
        start_time = int(bin_label.split("-")[0])
        bin_center = start_time + bin_duration/2
        # Get the maximum score in this bin for annotation
        y_max = sub["Score"].max()
        y_min = sub["Score"].min()
        y_ann = y_max + 0.1*(y_max - y_min if y_max != y_min else 0.5)
        ax.text(bin_center, y_ann, f"p = {p_val:.3f}", ha="center", va="bottom", fontsize=10)
        if not np.isnan(p_val) and p_val < 0.05:
            ax.text(bin_center, y_ann + 0.05, "*", ha="center", va="bottom", fontsize=16, color="black")
    
    # Optionally, restrict the y-axis to a fixed range (or use percentiles)
    # Here we use a fixed range as an example:
    ax.set_ylim(-3, 3)
    
    # Format the plot
    ax.set_title(f"BF Scores Across 30-s Bins — Experiment {exp_label}", fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("BF Score", fontsize=14)
    ax.set_xticks(bin_centers)
    # Create x-tick labels like "0-30", "30-60", etc.
    xtick_labels = [f"{int(bin_label.split('-')[0])}-{int(bin_label.split('-')[1])}" for bin_label in unique_bins]
    ax.set_xticklabels(xtick_labels, fontsize=12)
    ax.legend(fontsize=12, title="Group")
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st



#%% DYNAMICAL NORMALIZED BFL ANALYSIS. each bf score is normalized agains the baseline instead of against
# the data from their own bin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

#%% DYNAMICAL NORMALIZED BFL ANALYSIS

# --- Set parameters ---
frame_rate = 25        # frames per second
bin_duration = 30     # seconds per bin
bin_size = frame_rate * bin_duration  # 25*30 = 750 frames per bin

# Define baseline period: first 3 minutes (180 seconds)
baseline_duration = 60
baseline_frames = frame_rate * baseline_duration  # 25*1800 = 45000 frames

# Define group colors
color_control = "#f9c74f"  # Control
color_els = "#c37bac"      # ELS

# Process each experiment ("1" and "3") separately
for exp_label in ["1", "3"]:
    # Get recordings for this experiment from df
    recs_exp = df[df["experiment"] == exp_label]["recording"].unique()
    
    ### Step 1: Compute baseline transition matrices for each recording
    baseline_trans = {}
    for rec in recs_exp:
        if rec not in results_dict or "syllable" not in results_dict[rec]:
            continue
        seq = results_dict[rec]["syllable"]
        if len(seq) >= baseline_frames:
            baseline_seq = seq[:baseline_frames]
            TM_baseline = compute_transition_matrix(baseline_seq)
            TM_baseline = TM_baseline.reindex(index=behavior_codes, columns=behavior_codes, fill_value=0)
            baseline_trans[rec] = TM_baseline
    
    ### Step 2: Compute group baseline medians (fixed reference)
    baseline_control_list = [baseline_trans[rec] for rec in baseline_trans if group_assignments[rec] == "Control"]
    baseline_els_list = [baseline_trans[rec] for rec in baseline_trans if group_assignments[rec] == "ELS"]
    median_baseline_control = compute_group_median(baseline_control_list)
    #median_baseline_control = compute_group_mean(baseline_control_list)
    median_baseline_els = compute_group_median(baseline_els_list)
    #median_baseline_els = compute_group_mean(baseline_els_list)
    
    ### Step 3: Split recordings into 30-s bins and compute each bin's transition matrix
    bins_dict = {}
    for rec in recs_exp:
        if rec not in results_dict or "syllable" not in results_dict[rec]:
            continue
        seq = results_dict[rec]["syllable"]
        num_bins = len(seq) // bin_size  # only complete bins
        for bin_idx in range(num_bins):
            start = bin_idx * bin_size
            end = (bin_idx + 1) * bin_size
            bin_seq = seq[start:end]
            TM_bin = compute_transition_matrix(bin_seq)
            TM_bin = TM_bin.reindex(index=behavior_codes, columns=behavior_codes, fill_value=0)
            if bin_idx not in bins_dict:
                bins_dict[bin_idx] = {"trans_mats": {}, "group_assignments": {}}
            bins_dict[bin_idx]["trans_mats"][rec] = TM_bin
            bins_dict[bin_idx]["group_assignments"][rec] = group_assignments[rec]
    
    ### Step 4: For each bin, compute the normalized BF score relative to the fixed baseline medians.
    eps = 1e-5  # small constant to avoid division by zero
    bf_data = []
    for bin_idx, bin_data in bins_dict.items():
        time_bin_label = f"{bin_idx*30}-{(bin_idx+1)*30}"
        for rec, TM_bin in bin_data["trans_mats"].items():
            # Compute Manhattan distances to the fixed baseline medians
            d_control = manhattan_distance(TM_bin, median_baseline_control)
            d_els = manhattan_distance(TM_bin, median_baseline_els)
            norm_score = np.log((d_control + eps) / (d_els + eps))
            bf_data.append({"Time_Bin": time_bin_label,
                            "Bin_Start": bin_idx*30,  # numeric start time
                            "Group": bin_data["group_assignments"][rec],
                            "Score": norm_score})
    
    ### Step 5: Aggregate normalized BF scores into a DataFrame
    df_bf = pd.DataFrame(bf_data)
    df_bf = df_bf.sort_values("Bin_Start")
    unique_bins = df_bf["Time_Bin"].unique()
    
    ### Step 6: Compute group mean and SEM per bin
    stats_list = []
    for bin_label in unique_bins:
        for grp in ["Control", "ELS"]:
            sub = df_bf[(df_bf["Time_Bin"] == bin_label) & (df_bf["Group"] == grp)]
            mean_val = sub["Score"].mean()
            sem_val = sub["Score"].sem() if len(sub) > 1 else np.nan
            start_time = int(bin_label.split("-")[0])
            center_time = start_time + bin_duration/2
            stats_list.append({"Time_Bin": bin_label, "Bin_Center": center_time,
                                "Group": grp, "Mean": mean_val, "SEM": sem_val})
    df_stats = pd.DataFrame(stats_list)
    
    ### Step 7: For each time bin, perform a t-test between Control and ELS
    ttest_results = {}
    for bin_label in unique_bins:
        sub = df_bf[df_bf["Time_Bin"] == bin_label]
        control_scores = sub[sub["Group"] == "Control"]["Score"].values
        els_scores = sub[sub["Group"] == "ELS"]["Score"].values

        if len(control_scores) > 1 and len(els_scores) > 1:
            t_stat, p_val = st.ttest_ind(control_scores, els_scores, equal_var=False)
        else:
            p_val = np.nan
        ttest_results[bin_label] = p_val
    
    ### Step 8: Plot the dynamic evolution (line plot with mean & SEM shaded, individual dots, and p-value annotations)
    fig, ax = plt.subplots(figsize=(12, 6))
    bin_centers = sorted(df_stats["Bin_Center"].unique())
    
    # Plot mean lines and SEM shaded for each group
    for grp, col in zip(["Control", "ELS"], [color_control, color_els]):
        df_grp = df_stats[df_stats["Group"] == grp].sort_values("Bin_Center")
        x_vals = df_grp["Bin_Center"].values
        y_vals = df_grp["Mean"].values
        y_err = df_grp["SEM"].values
        ax.plot(x_vals, y_vals, marker="o", color=col, label=grp, linewidth=2)
        ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err, color=col, alpha=0.3)
    
    # Overlay individual data points with horizontal offsets for clarity
    for grp, col in zip(["Control", "ELS"], [color_control, color_els]):
        df_grp = df_bf[df_bf["Group"] == grp]
        offset = -3 if grp == "Control" else 3  # fixed offset in seconds
        jitter = np.random.uniform(-1, 1, size=len(df_grp))
        ax.scatter(df_grp["Bin_Start"] + bin_duration/2 + offset + jitter, df_grp["Score"],
                   color=col, alpha=0.8, s=60, zorder=5)
    
    # Annotate each time bin with its p-value and a significance star if p < 0.05
    for bin_label in unique_bins:
        sub = df_bf[df_bf["Time_Bin"] == bin_label]
        p_val = ttest_results[bin_label]
        start_time = int(bin_label.split("-")[0])
        bin_center = start_time + bin_duration/2
        y_max = sub["Score"].max()
        y_min = sub["Score"].min()
        y_ann = y_max + 0.1 * (y_max - y_min if y_max != y_min else 0.5)
        ax.text(bin_center, y_ann, f"p = {p_val:.3f}", ha="center", va="bottom", fontsize=10)
        if not np.isnan(p_val) and p_val < 0.05:
            ax.text(bin_center, y_ann + 0.05, "*", ha="center", va="bottom", fontsize=16, color="black")
    
    # Optionally, restrict the y-axis (here a fixed range is set as an example)
    #ax.set_ylim(-3, 3)
    
    # Format the plot
    ax.set_title(f"Normalized BF Scores (Baseline: First 30 min) Across 30-s Bins — Experiment {exp_label}", fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Normalized BF Score", fontsize=14)
    ax.set_xticks(bin_centers)
    xtick_labels = [f"{int(bin_label.split('-')[0])}-{int(bin_label.split('-')[1])}" for bin_label in unique_bins]
    ax.set_xticklabels(xtick_labels, fontsize=12)
    ax.legend(fontsize=12, title="Group")
    plt.tight_layout()
    plt.show()



#%% DYNAMIC ALL EXPERIMENTS TOGETHER

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

#%% DYNAMICAL NORMALIZED BFL ANALYSIS (Combined Experiments)

# --- Set parameters ---
frame_rate = 25         # frames per second
bin_duration = 30      # seconds per bin (i.e. each bin is 180 sec)
bin_size = frame_rate * bin_duration  # 25 * 180 = 4500 frames per bin

# Define baseline period: first 3 minutes (180 seconds)
baseline_duration = 180  
baseline_frames = frame_rate * baseline_duration  # 25 * 180 = 4500 frames

# Define group colors
color_control = "#f9c74f"  # Control
color_els = "#c37bac"      # ELS

### Process BOTH experiments together
# Get recordings for experiments "1" and "3" from df
recs_all = df[df["experiment"].isin(["1", "3"])]["recording"].unique()

### Step 1: Compute baseline transition matrices for each recording
baseline_trans = {}
for rec in recs_all:
    if rec not in results_dict or "syllable" not in results_dict[rec]:
        continue
    seq = results_dict[rec]["syllable"]
    if len(seq) >= baseline_frames:
        baseline_seq = seq[:baseline_frames]
        TM_baseline = compute_transition_matrix(baseline_seq)
        TM_baseline = TM_baseline.reindex(index=behavior_codes, columns=behavior_codes, fill_value=0)
        baseline_trans[rec] = TM_baseline

### Step 2: Compute group baseline medians (fixed reference)
baseline_control_list = [baseline_trans[rec] for rec in baseline_trans if group_assignments[rec] == "Control"]
baseline_els_list = [baseline_trans[rec] for rec in baseline_trans if group_assignments[rec] == "ELS"]
median_baseline_control = compute_group_median(baseline_control_list)
median_baseline_els = compute_group_median(baseline_els_list)

### Step 3: Split recordings into bins and compute each bin's transition matrix
bins_dict = {}
for rec in recs_all:
    if rec not in results_dict or "syllable" not in results_dict[rec]:
        continue
    seq = results_dict[rec]["syllable"]
    num_bins = len(seq) // bin_size  # only complete bins
    for bin_idx in range(num_bins):
        start = bin_idx * bin_size
        end = (bin_idx + 1) * bin_size
        bin_seq = seq[start:end]
        TM_bin = compute_transition_matrix(bin_seq)
        TM_bin = TM_bin.reindex(index=behavior_codes, columns=behavior_codes, fill_value=0)
        if bin_idx not in bins_dict:
            bins_dict[bin_idx] = {"trans_mats": {}, "group_assignments": {}}
        bins_dict[bin_idx]["trans_mats"][rec] = TM_bin
        bins_dict[bin_idx]["group_assignments"][rec] = group_assignments[rec]

### Step 4: For each bin, compute the normalized BF score relative to the fixed baseline medians.
eps = 1e-5  # small constant to avoid division by zero
bf_data = []
for bin_idx, bin_data in bins_dict.items():
    time_bin_label = f"{bin_idx*180}-{(bin_idx+1)*180}"
    for rec, TM_bin in bin_data["trans_mats"].items():
        # Compute Manhattan distances to the fixed baseline medians
        d_control = manhattan_distance(TM_bin, median_baseline_control)
        d_els = manhattan_distance(TM_bin, median_baseline_els)
        norm_score = np.log((d_control + eps) / (d_els + eps))
        bf_data.append({
            "Time_Bin": time_bin_label,
            "Bin_Start": bin_idx * 180,  # numeric start time in seconds
            "Group": bin_data["group_assignments"][rec],
            "Score": norm_score
        })

### Step 5: Aggregate normalized BF scores into a DataFrame
df_bf = pd.DataFrame(bf_data)
df_bf = df_bf.sort_values("Bin_Start")
unique_bins = df_bf["Time_Bin"].unique()

### Step 6: Compute group mean and SEM per bin
stats_list = []
for bin_label in unique_bins:
    for grp in ["Control", "ELS"]:
        sub = df_bf[(df_bf["Time_Bin"] == bin_label) & (df_bf["Group"] == grp)]
        mean_val = sub["Score"].mean()
        sem_val = sub["Score"].sem() if len(sub) > 1 else np.nan
        start_time = int(bin_label.split("-")[0])
        center_time = start_time + bin_duration / 2
        stats_list.append({
            "Time_Bin": bin_label,
            "Bin_Center": center_time,
            "Group": grp,
            "Mean": mean_val,
            "SEM": sem_val
        })
df_stats = pd.DataFrame(stats_list)

### Step 7: For each time bin, perform a t-test between Control and ELS
ttest_results = {}
for bin_label in unique_bins:
    sub = df_bf[df_bf["Time_Bin"] == bin_label]
    control_scores = sub[sub["Group"] == "Control"]["Score"].values
    els_scores = sub[sub["Group"] == "ELS"]["Score"].values
    if len(control_scores) > 1 and len(els_scores) > 1:
        t_stat, p_val = st.ttest_ind(control_scores, els_scores, equal_var=False)
    else:
        p_val = np.nan
    ttest_results[bin_label] = p_val

### Step 8: Plot the dynamic evolution (line plot with mean & SEM shaded, individual dots, and p-value annotations)
fig, ax = plt.subplots(figsize=(12, 6))
bin_centers = sorted(df_stats["Bin_Center"].unique())

# Plot mean lines and SEM shaded for each group
for grp, col in zip(["Control", "ELS"], [color_control, color_els]):
    df_grp = df_stats[df_stats["Group"] == grp].sort_values("Bin_Center")
    x_vals = df_grp["Bin_Center"].values
    y_vals = df_grp["Mean"].values
    y_err = df_grp["SEM"].values
    ax.plot(x_vals, y_vals, marker="o", color=col, label=grp, linewidth=2)
    ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err, color=col, alpha=0.3)

# Overlay individual data points with horizontal offsets for clarity
for grp, col in zip(["Control", "ELS"], [color_control, color_els]):
    df_grp = df_bf[df_bf["Group"] == grp]
    offset = -3 if grp == "Control" else 3  # fixed offset (in seconds)
    jitter = np.random.uniform(-1, 1, size=len(df_grp))
    ax.scatter(df_grp["Bin_Start"] + bin_duration/2 + offset + jitter, df_grp["Score"],
               color=col, alpha=0.8, s=60, zorder=5)

# Annotate each time bin with its p-value and a significance star if p < 0.05
for bin_label in unique_bins:
    sub = df_bf[df_bf["Time_Bin"] == bin_label]
    p_val = ttest_results[bin_label]
    start_time = int(bin_label.split("-")[0])
    bin_center = start_time + bin_duration / 2
    y_max = sub["Score"].max()
    y_min = sub["Score"].min()
    y_ann = y_max + 0.1 * (y_max - y_min if y_max != y_min else 0.5)
    ax.text(bin_center, y_ann, f"p = {p_val:.3f}", ha="center", va="bottom", fontsize=10)
    if not np.isnan(p_val) and p_val < 0.05:
        ax.text(bin_center, y_ann + 0.05, "*", ha="center", va="bottom", fontsize=16, color="black")

# Optionally, restrict the y-axis (here you can set a fixed range or use percentiles)
# ax.set_ylim(-3, 3)

# Format the plot
ax.set_title("Normalized BF Scores (Baseline: First 3 min) Across 3-min Bins — Combined Experiments", fontsize=16)
ax.set_xlabel("Time (s)", fontsize=14)
ax.set_ylabel("Normalized BF Score", fontsize=14)
ax.set_xticks(bin_centers)
xtick_labels = [f"{int(bin_label.split('-')[0])}-{int(bin_label.split('-')[1])}" for bin_label in unique_bins]
ax.set_xticklabels(xtick_labels, fontsize=12)
ax.legend(fontsize=12, title="Group")
plt.tight_layout()
plt.show()


#%% PHASES

# Define intervals (in seconds)
exploration = [[0, 180]]
trial = [[180, 180 + 30], [270, 270 + 30], [360, 360 + 30]]
post_trial = [[180 + 30, 180 + 30 + 60], [270 + 30, 270 + 30 + 60], [360 + 30, 360 + 30 + 60]]

fig, ax = plt.subplots(figsize=(10, 2))

# Shade exploration area
for i, (start, end) in enumerate(exploration):
    ax.axvspan(start, end, color='skyblue', alpha=0.5,
               label='Exploration' if i == 0 else None)

# Shade trial areas
for i, (start, end) in enumerate(trial):
    ax.axvspan(start, end, color='limegreen', alpha=0.5,
               label='Trial' if i == 0 else None)

# Shade post-trial areas
for i, (start, end) in enumerate(post_trial):
    ax.axvspan(start, end, color='salmon', alpha=0.5,
               label='Post-Trial' if i == 0 else None)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Activity')
ax.set_title('Shaded Areas for Exploration, Trial, and Post-Trial')
ax.set_xlim(0, 450)
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

#%%

# --- Define phase intervals (in seconds) ---
exploration_intervals = [[0, 180]]                         # Exploration: 0–180 s
trial_intervals       = [[180, 210], [270, 300], [360, 390]] # Trial: three 30-s intervals
inter_trial_intervals = [[210, 270], [300, 360], [390, 450]] # Inter-trial (post-trial): three 60-s intervals

# --- Helper: Compute summed transition matrix for given intervals ---
def compute_phase_transition_matrix(seq, intervals, frame_rate=25):
    """
    For a given behavior sequence (seq) and a list of intervals (in seconds),
    compute the summed transition matrix (using compute_transition_matrix).
    """
    phase_tm = None
    for start_sec, end_sec in intervals:
        start_frame = int(start_sec * frame_rate)
        end_frame   = int(end_sec * frame_rate)
        if len(seq) < end_frame:
            continue  # skip this interval if the recording is too short
        seg = seq[start_frame:end_frame]
        TM = compute_transition_matrix(seg)
        TM = TM.reindex(index=behavior_codes, columns=behavior_codes, fill_value=0)
        if phase_tm is None:
            phase_tm = TM.copy()
        else:
            phase_tm += TM
    return phase_tm

# --- Set up phase labels and corresponding intervals ---
phases = ['Exploration', 'Trial', 'InterTrial']
phase_intervals = {
    'Exploration': exploration_intervals,
    'Trial': trial_intervals,
    'InterTrial': inter_trial_intervals
}

# --- Compute phase-specific transition matrices and record group assignments ---
phase_trans_mats = {phase: {} for phase in phases}
phase_group_assignments = {phase: {} for phase in phases}

for rec, data in results_dict.items():
    if rec not in group_assignments or 'syllable' not in data:
        continue
    seq = data['syllable']
    for phase in phases:
        tm_phase = compute_phase_transition_matrix(seq, phase_intervals[phase], frame_rate=25)
        if tm_phase is not None:
            phase_trans_mats[phase][rec] = tm_phase
            phase_group_assignments[phase][rec] = group_assignments[rec]

# --- Compute BF (BFL) scores for each phase using your existing function ---
phase_bfl_scores = {}
for phase in phases:
    try:
        # compute_bfl_scores expects a dictionary of transition matrices and matching group assignments.
        bfl_phase = compute_bfl_scores(phase_trans_mats[phase], phase_group_assignments[phase], 'Control', 'ELS')
        phase_bfl_scores[phase] = bfl_phase
    except Exception as e:
        print(f"Error computing BF scores for phase {phase}: {e}")

# --- Combine BF scores into a DataFrame for plotting ---
bf_data = []
for phase in phases:
    if phase in phase_bfl_scores:
        for rec, score in phase_bfl_scores[phase].items():
            bf_data.append({
                'Recording': rec,
                'Phase': phase,
                'BFL_Score': score,
                'Group': phase_group_assignments[phase][rec]
            })
df_bf_phase = pd.DataFrame(bf_data)

# --- Plot the results: Boxplots with overlaid scatter (swarm) ---
plt.figure(figsize=(8, 6))
# Boxplot by Phase and Group
sns.boxplot(x='Phase', y='BFL_Score', hue='Group', data=df_bf_phase,
            palette={'Control': 'skyblue', 'ELS': 'salmon'},
            fliersize=0)  # hide outliers from boxplot (they will be shown in swarmplot)
# Overlay individual data points (with dodge to separate groups)
sns.swarmplot(x='Phase', y='BFL_Score', hue='Group', data=df_bf_phase,
              dodge=True, color='black', alpha=0.8, size=6)
# Remove duplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
n = len(df_bf_phase['Group'].unique())
plt.legend(handles[:n], labels[:n], title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("Phase")
plt.ylabel("BFL Score (log(distance_Control/distance_ELS))")
plt.title("Behavioral Flow Scores by Phase and Group")
plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

# --- Phase Definitions (in seconds) ---
# Exploration: first 180 seconds
exploration_intervals = [[0, 180]]
# Trial: three 30-s periods at 180-210, 270-300, 360-390 s
trial_intervals       = [[180, 210], [270, 300], [360, 390]]
# Inter-trial: the gaps between trials (here 210-270, 300-360, 390-450 s)
inter_trial_intervals = [[210, 270], [300, 360], [390, 450]]

phases = ['Exploration', 'Trial', 'InterTrial']
phase_intervals = {
    'Exploration': exploration_intervals,
    'Trial': trial_intervals,
    'InterTrial': inter_trial_intervals
}

# --- Helper function: Compute the summed transition matrix for given phase intervals ---
def compute_phase_transition_matrix(seq, intervals, frame_rate=25):
    """
    Given a behavior sequence (seq) and a list of time intervals (in seconds),
    compute the summed transition matrix for that phase.
    """
    phase_tm = None
    for start_sec, end_sec in intervals:
        start_frame = int(start_sec * frame_rate)
        end_frame   = int(end_sec * frame_rate)
        if len(seq) < end_frame:
            continue  # skip this interval if the recording is too short
        seg = seq[start_frame:end_frame]
        TM = compute_transition_matrix(seg)
        TM = TM.reindex(index=behavior_codes, columns=behavior_codes, fill_value=0)
        if phase_tm is None:
            phase_tm = TM.copy()
        else:
            phase_tm += TM
    return phase_tm

# --- Loop over experiments and phases to compute BF scores ---
# We'll accumulate data in a list of dicts.
bf_data_all = []
# For each experiment label ("1" and "3") from your combined DataFrame
for exp_label in ["1", "3"]:
    # Get recordings for this experiment from df
    recs_exp = df[df["experiment"] == exp_label]["recording"].unique()
    for phase in phases:
        # Dictionaries for this phase
        phase_trans_mats = {}
        phase_group_assignments = {}
        for rec in recs_exp:
            # Make sure the recording exists and has a 'syllable' array
            if rec not in results_dict or "syllable" not in results_dict[rec]:
                continue
            seq = results_dict[rec]["syllable"]
            tm_phase = compute_phase_transition_matrix(seq, phase_intervals[phase], frame_rate=25)
            if tm_phase is not None:
                phase_trans_mats[rec] = tm_phase
                phase_group_assignments[rec] = group_assignments[rec]
        if not phase_trans_mats:
            continue  # skip if no data for this phase
        try:
            # Compute BF (BFL) scores using your function
            bfl_scores_phase = compute_bfl_scores(phase_trans_mats, phase_group_assignments, 'Control', 'ELS')
        except Exception as e:
            print(f"Error computing BF scores for experiment {exp_label}, phase {phase}: {e}")
            continue
        # Append each recording's BF score with its experiment, phase, and group info
        for rec, score in bfl_scores_phase.items():
            bf_data_all.append({
                "Recording": rec,
                "Experiment": exp_label,
                "Phase": phase,
                "BFL_Score": score,
                "Group": phase_group_assignments[rec]
            })

# Convert to DataFrame for plotting and stats
df_bf = pd.DataFrame(bf_data_all)

# --- Plotting & Statistical Tests Per Experiment ---
for exp_label in ["1", "3"]:
    df_exp = df_bf[df_bf["Experiment"] == exp_label]
    plt.figure(figsize=(10, 6))
    
    # Create boxplots (without outlier markers) by Phase and Group.
    sns.boxplot(x='Phase', y='BFL_Score', hue='Group', data=df_exp,
                palette={'Control': 'skyblue', 'ELS': 'salmon'}, fliersize=0)
    # Overlay individual data points using a swarmplot (with dodge)
    sns.swarmplot(x='Phase', y='BFL_Score', hue='Group', data=df_exp,
                  dodge=True, color='black', alpha=0.8, size=6)
    
    # Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    n = len(df_exp["Group"].unique())
    plt.legend(handles[:n], labels[:n], title="Group", loc='upper right')
    
    plt.title(f"BFL Scores by Phase and Group (Experiment {exp_label})", fontsize=14)
    plt.xlabel("Phase", fontsize=12)
    plt.ylabel("BFL Score (log(dA/dB))", fontsize=12)
    
    # --- For each phase, perform a t-test comparing groups and annotate the p-value ---
    for phase in phases:
        sub = df_exp[df_exp["Phase"] == phase]
        control_scores = sub[sub["Group"] == "Control"]["BFL_Score"].values
        els_scores = sub[sub["Group"] == "ELS"]["BFL_Score"].values
        if len(control_scores) > 1 and len(els_scores) > 1:
            t_stat, p_val = st.ttest_ind(control_scores, els_scores, equal_var=False)
        else:
            p_val = np.nan
        # Use the x-position corresponding to the phase (assumes phases are ordered as in the list)
        phase_idx = phases.index(phase)
        # For y-position, take the maximum score in that phase (with a small vertical offset)
        y_max = sub["BFL_Score"].max() if not sub.empty else 0
        plt.text(phase_idx, y_max + 0.1*(abs(y_max) if y_max != 0 else 1),
                 f"p = {p_val:.3f}", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
