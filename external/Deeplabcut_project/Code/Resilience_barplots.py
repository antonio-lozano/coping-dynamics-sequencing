# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:19:27 2025

@author: jsangui
"""
#%% Imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from numba import njit
from scipy.spatial.distance import jensenshannon, correlation
import pandas as pd
from scipy.stats import ttest_ind
from adjustText import adjust_text
from collections import Counter, defaultdict
import statsmodels.formula.api as smf

#%% Build Feature Matrix from Syllable Sequence (per Animal x Experiment)
# Here we group by both Animal and Experiment.
animal_exp_freq = {}
all_syllables_set = set()

for animal, df in data.items():
    # Group by Experiment in case an animal participated in >1 experiment.
    for exp, df_exp in df.groupby("Experiment"):
        df_exp = df_exp.sort_values("Time_bin")
        # Count syllable occurrences.
        # (If you prefer weighted counts, replace the following with a weighted sum using row["Percentage"].)
        counts = Counter(df_exp["Syllable"])
        condition = df_exp.iloc[0]["Condition"]
        animal_exp_freq[(animal, exp)] = {"counts": counts, "Condition": condition}
        all_syllables_set.update(counts.keys())

all_syllables = sorted(list(all_syllables_set))
print(f"Unique syllables found: {all_syllables}")

# Build the feature matrix: one row per (Animal, Experiment)
feature_keys = list(animal_exp_freq.keys())
n_entries = len(feature_keys)
X = np.zeros((n_entries, len(all_syllables)))
animal_labels = []      # for later use in plotting and modeling
experiment_labels = []
condition_labels = []

for i, key in enumerate(feature_keys):
    counts = animal_exp_freq[key]["counts"]
    for j, syl in enumerate(all_syllables):
        X[i, j] = counts.get(syl, 0)
    # Normalize the frequency vector if not all zeros.
    if X[i].sum() > 0:
        X[i] /= X[i].sum()
    animal_labels.append(key[0])
    experiment_labels.append(key[1])
    condition_labels.append(animal_exp_freq[key]["Condition"])

print("Feature matrix computed.")

#%% Compute Distance Matrices from the Feature Matrix
print("Computing distance matrices...")

# 1. Cosine Distance
norms = np.linalg.norm(X, axis=1)
dot = X @ X.T
dist_cosine = 1 - (dot / (norms[:, None] * norms[None, :]))

# 2. Jensen–Shannon Distance
dist_js = np.zeros((n_entries, n_entries))
for i in range(n_entries):
    for j in range(i + 1, n_entries):
        d = jensenshannon(X[i], X[j])
        dist_js[i, j] = dist_js[j, i] = d

# 3. Manhattan Distance
dist_manhattan = np.abs(X[:, None, :] - X[None, :, :]).sum(axis=2)

# 4. Correlation Distance
dist_corr = np.zeros((n_entries, n_entries))
for i in range(n_entries):
    for j in range(i + 1, n_entries):
        d = correlation(X[i], X[j])
        dist_corr[i, j] = dist_corr[j, i] = d

# 5. Euclidean Distance
dist_euclidean = np.sqrt(((X[:, None, :] - X[None, :, :])**2).sum(axis=2))

distance_matrices = {
    "Cosine": dist_cosine,
    "Jensen–Shannon": dist_js,
    "Manhattan": dist_manhattan,
    "Correlation": dist_corr,
    "Euclidean": dist_euclidean,
}

print("Distance matrices computed.\n")

#%% MDS Scatter Plots (Color-coded by Condition)
print("Generating MDS scatter plots...")
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs = axs.flatten()
num_plots = len(distance_matrices)

for i, (name, dmat) in enumerate(distance_matrices.items()):
    ax = axs[i]
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(dmat)
    
    for j in range(n_entries):
        # Choose color based on condition.
        col = "#c37ba0" if condition_labels[j] == "ELS" else "#f9c74f"
        ax.scatter(coords[j, 0], coords[j, 1], s=200, color=col, edgecolor=col)
        label = f"{animal_labels[j]}_{experiment_labels[j]}"
        ax.text(coords[j, 0], coords[j, 1], label, fontsize=10, color='#4d4d4d',
                ha="center", va="center")
    
    ax.set_title(f"MDS ({name} Distance)", fontsize=14, color="#4d4d4d")
    ax.set_xlabel("Dim 1", fontsize=12, color="#4d4d4d")
    ax.set_ylabel("Dim 2", fontsize=12, color="#4d4d4d")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#4d4d4d")
    ax.tick_params(axis="both", colors="#4d4d4d")
    
    # Add legend for conditions.
    legend_labels = ["Control", "ELS"]
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#f9c74f", markeredgecolor="#f9c74f", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#c37ba0", markeredgecolor="#c37ba0", markersize=10)
    ]
    leg = ax.legend(handles, legend_labels, loc="lower right", frameon=False,
                    fontsize=10, ncol=2, columnspacing=0.5, handlelength=1)
    for text in leg.get_texts():
        text.set_color('#4d4d4d')

for idx in range(num_plots, len(axs)):
    fig.delaxes(axs[idx])

plt.tight_layout()
plt.savefig("MDS_syllable_sequence.png", bbox_inches='tight', dpi=700, transparent=True)
plt.show()

#%% Helper Function: MDS Coordinates and Behavioral Score
# The behavioral score is defined as:
#   score = log( d(point, median_ELS) / d(point, median_Control) )
epsilon = 1e-8  # to avoid division by zero

def get_mds_coords_and_scores(dmat, epsilon=1e-8):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(dmat)
    idx_els = [j for j, cond in enumerate(condition_labels) if cond == "ELS"]
    idx_ctrl = [j for j, cond in enumerate(condition_labels) if cond == "Control"]
    median_els = np.median(coords[idx_els], axis=0)
    median_ctrl = np.median(coords[idx_ctrl], axis=0)
    scores = np.array([
        np.log((np.linalg.norm(coords[j] - median_els) + epsilon) /
               (np.linalg.norm(coords[j] - median_ctrl) + epsilon))
        for j in range(len(condition_labels))
    ])
    return coords, scores, median_els, median_ctrl

#%% Frequency Behavioral Score Boxplots with t-Test
print("Generating frequency behavioral score boxplots...")
fig2, axs2 = plt.subplots(1, len(distance_matrices), figsize=(5 * len(distance_matrices), 5))
if len(distance_matrices) == 1:
    axs2 = [axs2]

for i, (name, dmat) in enumerate(distance_matrices.items()):
    print(f"Processing {name} distance...")
    _, scores, _, _ = get_mds_coords_and_scores(dmat, epsilon)
    
    # Split scores by condition.
    scores_ctrl = scores[np.array([cond == "Control" for cond in condition_labels])]
    scores_els  = scores[np.array([cond == "ELS" for cond in condition_labels])]
    
    # Two-sample t-test.
    stat, p_value = ttest_ind(scores_els, scores_ctrl, equal_var=False)
    print(f"  {name} t-test: p-value = {p_value:.3g}")
    
    data_to_plot = [scores_ctrl, scores_els]
    ax = axs2[i]
    bp = ax.boxplot(data_to_plot, positions=[1, 2], widths=0.4, patch_artist=True,
                    showfliers=False, medianprops={'linewidth': 3})
    
    group_colors = ["#f9c74f", "#c37ba0"]
    for patch, col in zip(bp['boxes'], group_colors):
        patch.set_facecolor("none")
        patch.set_edgecolor(col)
        patch.set_linewidth(2)
    for j in range(len(group_colors)):
        bp['whiskers'][2*j].set_color(group_colors[j])
        bp['whiskers'][2*j+1].set_color(group_colors[j])
        bp['whiskers'][2*j].set_linewidth(2)
        bp['whiskers'][2*j+1].set_linewidth(2)
    for j in range(len(group_colors)):
        bp['caps'][2*j].set_color(group_colors[j])
        bp['caps'][2*j+1].set_color(group_colors[j])
    for patch, col in zip(bp['medians'], group_colors):
        patch.set_color(col)
    for flier in bp['fliers']:
        flier.set(markerfacecolor=group_colors[0], markeredgecolor=group_colors[0])
    
    # Overlay jittered dots.
    jitter = 0.08
    for group_idx, group in enumerate(data_to_plot):
        x = np.random.normal(1 + group_idx, jitter, size=len(group))
        ax.scatter(x, group, color=group_colors[group_idx], edgecolor=group_colors[group_idx],
                   zorder=2, alpha=0.8)
        
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Control", "ELS"], fontsize=12, color="#4d4d4d")
    ax.set_ylabel("Frequency Behavioral Score\nlog(d(point, median ELS) / d(point, median Control))",
                  fontsize=12, color="#4d4d4d")
    ax.set_title(f"{name} Score", fontsize=14, color="#4d4d4d")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#4d4d4d")
    ax.tick_params(axis='both', colors='#4d4d4d')
    print(f"  Boxplot created for {name}.\n")

if len(distance_matrices) < len(axs2):
    for j in range(len(distance_matrices), len(axs2)):
        fig2.delaxes(axs2[j])
        
plt.tight_layout()
plt.savefig("Boxplots_MDS_syllable_sequence.png", bbox_inches='tight', dpi=700, transparent=True)
plt.show()

#%% Mixed Model Analysis using Frequency Behavioral Score
print("\nRunning mixed model analyses for each distance method...")
mixed_model_results = {}

for method_name, dmat in distance_matrices.items():
    print(f"\nProcessing mixed model for {method_name} distance...")
    _, scores, _, _ = get_mds_coords_and_scores(dmat, epsilon)
    
    # Build a DataFrame for modeling.
    df_model = pd.DataFrame({
        "Animal": animal_labels,
        "Experiment": experiment_labels,
        "Condition": condition_labels,
        "Score": scores
    })
    
    # Fit a mixed model with random intercept for Animal (animals may appear in multiple experiments).
    model = smf.mixedlm("Score ~ Condition + Experiment", df_model, 
                        groups=df_model["Animal"], 
                        vc_formula={"Experiment": "1"})
    result = model.fit(reml=False)
    
    print(f"\nMixed Model Results for {method_name} Distance:")
    print(result.summary())
    mixed_model_results[method_name] = result

#%% Aggregated Analysis (All Animals Combined)

# Define group conditions.
group1_conditions = ["ELS", "resilient_ELS"]
group2_conditions = ["Control", "vulnerable_controls"]

# Colors and plotting order.
group_colors = {"ELS": "#c37ba0", "Control": "#f9c74f"}
agg_group_labels = ["Control", "ELS"]
agg_group_positions = {"Control": 1, "ELS": 2}

# Use existing variables from the context.
animals = animal_labels            # each entry's animal ID
animal_conditions = condition_labels  # each entry's condition

# Build aggregated group assignment for each entry.
group_assignment_all = []
for cond in animal_conditions:
    if cond in group1_conditions:
        group_assignment_all.append("ELS")
    elif cond in group2_conditions:
        group_assignment_all.append("Control")
    else:
        group_assignment_all.append(None)

aggregated_results = {}

fig_agg, axs_agg = plt.subplots(1, len(distance_matrices), figsize=(5 * len(distance_matrices), 5))
if len(distance_matrices) == 1:
    axs_agg = [axs_agg]

for idx, (name, dmat) in enumerate(distance_matrices.items()):
    print(f"Aggregated analysis for {name} distance...")
    # Run MDS on the full distance matrix.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(dmat)
    
    # Compute medians for each aggregated group.
    indices_control = [i for i, grp in enumerate(group_assignment_all) if grp == "Control"]
    indices_els     = [i for i, grp in enumerate(group_assignment_all) if grp == "ELS"]
    median_control = np.median(coords[indices_control], axis=0) if indices_control else None
    median_els     = np.median(coords[indices_els], axis=0) if indices_els else None
    
    # Compute frequency behavioral scores:
    #    score = log( d(Animal, median_ELS) / d(Animal, median_Control) )
    scores = []
    for i in range(len(animals)):
        d_els     = np.linalg.norm(coords[i] - median_els) if median_els is not None else np.nan
        d_control = np.linalg.norm(coords[i] - median_control) if median_control is not None else np.nan
        score = np.log((d_els + epsilon) / (d_control + epsilon))
        scores.append(score)
    scores = np.array(scores)
    
    # Separate scores by group in the desired order.
    scores_control = [scores[i] for i, grp in enumerate(group_assignment_all) if grp == "Control"]
    scores_els     = [scores[i] for i, grp in enumerate(group_assignment_all) if grp == "ELS"]
    data_to_plot = [scores_control, scores_els]
    
    ax = axs_agg[idx]
    bp = ax.boxplot(
        data_to_plot,
        positions=[agg_group_positions["Control"], agg_group_positions["ELS"]],
        widths=0.4,
        patch_artist=True,
        showfliers=False,
        medianprops={'linewidth': 3}
    )
    
    # Style the boxes.
    for patch, lab in zip(bp['boxes'], agg_group_labels):
        patch.set_facecolor("none")
        patch.set_edgecolor(group_colors[lab])
        patch.set_linewidth(2)
    
    # Style whiskers, caps, and medians.
    for j, lab in enumerate(agg_group_labels):
        c = group_colors[lab]
        bp['whiskers'][2*j].set_color(c)
        bp['whiskers'][2*j+1].set_color(c)
        bp['whiskers'][2*j].set_linewidth(2)
        bp['whiskers'][2*j+1].set_linewidth(2)
        bp['caps'][2*j].set_color(c)
        bp['caps'][2*j+1].set_color(c)
    for patch, lab in zip(bp['medians'], agg_group_labels):
        patch.set_color(group_colors[lab])
    
    # Overlay individual points with jitter.
    jitter = 0.08
    for i, score in enumerate(scores):
        grp = group_assignment_all[i]
        if grp is None:
            continue
        orig_cond = animal_conditions[i]
        # For ELS-like animals, expected score < 0; for Control-like, expected score > 0.
        if orig_cond in group1_conditions:
            marker = 'o' if score < 0 else 'x'
        elif orig_cond in group2_conditions:
            marker = 'o' if score > 0 else 'x'
        else:
            marker = 'o'
        x_val = np.random.normal(agg_group_positions[grp], jitter)
        ax.scatter(
            x_val, score,
            color=group_colors[grp],
            edgecolor=group_colors[grp],
            marker=marker,
            s=80, zorder=2
        )
    
    ax.set_xticks([agg_group_positions["Control"], agg_group_positions["ELS"]])
    ax.set_xticklabels(["Control", "ELS"], fontsize=12, color="#4d4d4d")
    ax.set_ylabel("Frequency Behavioral Score\nlog(d(Animal, Median ELS) / d(Animal, Median Control))",
                  fontsize=12, color="#4d4d4d")
    ax.set_title(f"{name} Score", fontsize=14, color="#4d4d4d")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#4d4d4d")
    ax.tick_params(axis='both', colors='#4d4d4d')
    
    aggregated_results[name] = {"similar": None, "different": None}

plt.tight_layout()
plt.savefig("MDS_Aggregated_Boxplots.png", bbox_inches='tight', dpi=700, transparent=True)
plt.show()

print("Aggregated Analysis Results:")
for name, res in aggregated_results.items():
    print(f"Distance Metric: {name}")
    print("  Results: See plot for individual markers\n")

#%% Experiment-Specific Analysis (Aggregated Groups) ####JEEEN

# Extract unique experiments from the existing experiment_labels.
experiments = sorted(set(experiment_labels))
experiment_results = {}

for exp in experiments:
    # Get indices corresponding to the current experiment.
    exp_indices = [i for i, exp_label in enumerate(experiment_labels) if exp_label == exp]
    exp_animals = [animals[i] for i in exp_indices]
    exp_conditions = [animal_conditions[i] for i in exp_indices]
    
    # Build aggregated group assignment for this experiment.
    group_assignment = []
    for cond in exp_conditions:
        if cond in group1_conditions:
            group_assignment.append("ELS")
        elif cond in group2_conditions:
            group_assignment.append("Control")
        else:
            group_assignment.append(None)
    
    experiment_results[exp] = {}
    n_metrics = len(distance_matrices)
    fig_exp, axs_exp = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axs_exp = [axs_exp]
    
    for idx_dm, (name, dmat) in enumerate(distance_matrices.items()):
        # Extract the submatrix for entries in the current experiment.
        dmat_sub = dmat[np.ix_(exp_indices, exp_indices)]
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        coords = mds.fit_transform(dmat_sub)
        
        # Compute medians for aggregated groups within this experiment.
        indices_control = [i for i, grp in enumerate(group_assignment) if grp == "Control"]
        indices_els     = [i for i, grp in enumerate(group_assignment) if grp == "ELS"]
        median_control = np.median(coords[indices_control], axis=0) if indices_control else None
        median_els     = np.median(coords[indices_els], axis=0) if indices_els else None
        
        # Compute frequency behavioral scores.
        scores = []
        for k in range(len(exp_animals)):
            d1 = np.linalg.norm(coords[k] - median_els) if median_els is not None else np.nan
            d2 = np.linalg.norm(coords[k] - median_control) if median_control is not None else np.nan
            score = np.log((d1 + epsilon) / (d2 + epsilon))
            scores.append(score)
        scores = np.array(scores)
        
        # Separate scores by group.
        scores_control = [scores[i] for i, grp in enumerate(group_assignment) if grp == "Control"]
        scores_els     = [scores[i] for i, grp in enumerate(group_assignment) if grp == "ELS"]
        data_to_plot = [scores_control, scores_els]
        
        ax = axs_exp[idx_dm]
        bp = ax.boxplot(
            data_to_plot,
            positions=[agg_group_positions["Control"], agg_group_positions["ELS"]],
            widths=0.4,
            patch_artist=True,
            showfliers=False,
            medianprops={'linewidth': 3}
        )
        
        # Style the boxes.
        for patch, lab in zip(bp['boxes'], agg_group_labels):
            patch.set_facecolor("none")
            patch.set_edgecolor(group_colors[lab])
            patch.set_linewidth(2)
        
        # Style whiskers, caps, and medians.
        for j, lab in enumerate(agg_group_labels):
            c = group_colors[lab]
            bp['whiskers'][2*j].set_color(c)
            bp['whiskers'][2*j+1].set_color(c)
            bp['whiskers'][2*j].set_linewidth(2)
            bp['whiskers'][2*j+1].set_linewidth(2)
            bp['caps'][2*j].set_color(c)
            bp['caps'][2*j+1].set_color(c)
        for patch, lab in zip(bp['medians'], agg_group_labels):
            patch.set_color(group_colors[lab])
        
        # Overlay individual points with jitter and cross notation.
        jitter = 0.08
        for i, score in enumerate(scores):
            grp = group_assignment[i]
            if grp is None:
                continue
            orig_cond = exp_conditions[i]
            if orig_cond in group1_conditions:
                marker = 'o' if score < 0 else 'x'
            elif orig_cond in group2_conditions:
                marker = 'o' if score > 0 else 'x'
            else:
                marker = 'o'
            x_val = np.random.normal(agg_group_positions[grp], jitter)
            ax.scatter(
                x_val, score,
                color=group_colors[grp],
                edgecolor=group_colors[grp],
                marker=marker,
                s=80, zorder=2
            )
        
        ax.set_xticks([agg_group_positions["Control"], agg_group_positions["ELS"]])
        ax.set_xticklabels(["Control", "ELS"], fontsize=12, color="#4d4d4d")
        ax.set_ylabel("Frequency Behavioral Score\nlog(d(Animal, Median ELS) / d(Animal, Median Control))",
                      fontsize=12, color="#4d4d4d")
        ax.set_title(f"{name} Score", fontsize=14, color="#4d4d4d")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#4d4d4d")
        ax.tick_params(axis='both', colors='#4d4d4d')
        
        experiment_results[exp][name] = {"groups": None, "scores": None}
    
    plt.suptitle(f"Frequency Behavioral Scores for Experiment {exp}", fontsize=16, color="#4d4d4d")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"Experiment_{exp}_MDS_Boxplots.png", bbox_inches='tight', dpi=700, transparent=True)
    plt.show()
    
    print(f"Results for Experiment {exp}:")
    for name, res in experiment_results[exp].items():
        print(f"  Distance Metric: {name}")
        print("    See plot for individual markers\n")

print("Experiment-specific analysis complete.")

#%% =============================================================================
# PART 1: Update Group Labels
# =============================================================================

print("Group counts BEFORE update:")
original_counts = Counter(animal_conditions)
print(original_counts)

# For this example we use the aggregated results from the "Manhattan" distance metric.
different_agg = aggregated_results["Manhattan"]["different"]

# Update group labels in animal_conditions based on different_agg.
# Here we update *all* occurrences (comparing as strings to avoid floating-point issues)
for animal in different_agg:
    for i, a in enumerate(animals):
        if str(a) == str(animal):
            if animal_conditions[i] == "ELS":
                animal_conditions[i] = "Resilient_ELS"
            elif animal_conditions[i] == "Control":
                animal_conditions[i] = "Vulnerable_control"

# Helper function to find a matching key in data (compares as strings).
def get_data_key(animal_id, keys):
    for key in keys:
        if str(key) == str(animal_id):
            return key
    return None

# Update each animal's DataFrame with the new group label.
# Looping over unique animal IDs avoids duplicate updates.
for animal in set(animals):
    idx = animals.index(animal)
    key = get_data_key(animal, data.keys())
    if key is not None:
        data[key]["Condition"] = animal_conditions[idx]
    else:
        print(f"Animal {animal} not found in data keys!")

print("\nGroup counts AFTER update:")
updated_counts = Counter(animal_conditions)
print(updated_counts)

# Group animals by their updated condition.
group_animals_updated = defaultdict(list)
for animal in set(animals):
    idx = animals.index(animal)
    group_animals_updated[animal_conditions[idx]].append(animal)

# Print each group with its animals.
for condition, animal_list in group_animals_updated.items():
    print(f"{condition}: {animal_list}")
