# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 08:54:19 2025

@author: jsangui

Reorganized Analysis Script using New_condition:
  🔹 1. Shannon Entropy
  🔹 2. Evenness
  🔹 3. Simpson's Diversity Index
  🔹 4. Cumulative Usage Index (CUI)
  🔹 5. Mean Bout Duration – Overall
  🔹 6. Mean Bout Duration – Per Cluster

For each metric:
  📈 Two mixed‐effects models are run:
      - One with Control as intercept (default)
      - One with ELS as intercept (by releveling)
  📐 Cohen’s d is computed for three comparisons:
      - ELS vs Control
      - ELS_resilient vs Control
      - ELS vs ELS_resilient

NOTE: In all plots the group "ELS_resilient" is shown as a two‑line label:
     "ELS" on the first line and "resilient" on the second.
"""

#%% IMPORTS & DATA LOADING
import os, json, re, glob, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator
import moviepy.editor as mpy
import statsmodels.formula.api as smf

#%% LOAD THE DATASET & ASSIGN CLUSTERS
csv_path = r'C:\Users\jsangui\OneDrive - UvA\Others\Downloads\Moseq_final\CSVs\Syllable_per_timebin_final(250ms).csv'
json_path = r"C:\Users\jsangui\OneDrive - UvA\Others\Downloads\Moseq_final\Clustering\Clusters_jsons\Behavioral_clusters_a_mano_definitivo_no_mix_inaccurate.json"

moseq_df = pd.read_csv(csv_path)
moseq_df.rename(columns={'Time Bin': 'Time_bin'}, inplace=True)
print("Moseq_df head:")
print(moseq_df.head())

with open(json_path, 'r') as f:
    syllable_to_cluster = json.load(f)
print("JSON mapping:")
print(syllable_to_cluster)

def assign_cluster(syllable):
    try:
        s = int(syllable)
    except Exception:
        s = syllable
    for cluster, syllables in syllable_to_cluster.items():
        if s in syllables:
            return cluster
    return ""

moseq_df['Cluster'] = moseq_df['Syllable'].apply(assign_cluster)

#%% REFINE GROUPS: DEFINE NEW CONDITION MAPPING
# Define mappings for ELS_resilient and ELS animals
els_resilient = ['2.4', '15.4', '26.4', '43.3', '43.5', '49.6', 
                 '123.3', '134.2', '148.4', '150.3', '150.5','159.4']

els = ['15.6', '23.4', '25.1', '32.2', '32.4', '49.5', '55.3', '55.5', '56.3',
       '63.3', '69.1', '69.2', '69.3', '69.5', '73.1', '73.4', '75.1',
       '88.2', '88.3', '128.4', '133.3', '123.1', '134.4', '144.3', '144.5', 
       '153.4', '154.4', '159.3', '159.5']

def map_new_condition(row):
    if row['Animal'] in els_resilient:
        return 'ELS_resilient'
    elif row['Animal'] in els:
        return 'ELS'
    else:
        return row['Condition']  # Keep Control (or any other) as is

moseq_df['Animal'] = moseq_df['Animal'].astype(str)
moseq_df['New_condition'] = moseq_df.apply(map_new_condition, axis=1)

print("\nNew condition grouping:")
print(moseq_df[['Animal', 'Condition', 'New_condition']].drop_duplicates())
check_mapping = moseq_df.groupby(['Condition', 'New_condition'])['Animal'].nunique().reset_index()
print(check_mapping)

# Define new color palette for plots
plot_colors = {
    "Control": "#f9c74f",
    "ELS": "#c37ba0",
    "ELS_resilient": "#90be6d"
}

# Helper: transform "ELS_resilient" to two lines for X-axis labels
def transform_label(cond):
    return "ELS\nresilient" if cond == "ELS_resilient" else cond

# Helper: compute Cohen's d from descriptive stats (group means and pooled SD)
def compute_cohen_d(model, group1, group2, stats_df):
    mean1 = stats_df.loc[stats_df['New_condition'] == group1, 'mean'].values[0]
    mean2 = stats_df.loc[stats_df['New_condition'] == group2, 'mean'].values[0]
    n1 = stats_df.loc[stats_df['New_condition'] == group1, 'N'].values[0]
    n2 = stats_df.loc[stats_df['New_condition'] == group2, 'N'].values[0]
    sd1 = stats_df.loc[stats_df['New_condition'] == group1, 'sd'].values[0]
    sd2 = stats_df.loc[stats_df['New_condition'] == group2, 'sd'].values[0]
    pooled_sd = np.sqrt(((n1 - 1)*sd1**2 + (n2 - 1)*sd2**2) / (n1+n2-2))
    return (mean2 - mean1) / pooled_sd

#%% SECTION 1: Shannon Entropy
# 🧮 Functions: Compute frequency-based metrics using New_condition
def compute_frequency_metrics(df):
    freq_df = df.groupby(['Animal', 'Experiment', 'New_condition', 'Cluster']).size().reset_index(name='Count')
    pivot_df = freq_df.pivot_table(index=['Animal','New_condition', 'Experiment'],
                                   columns='Cluster',
                                   values='Count',
                                   fill_value=0)
    total_counts = pivot_df.sum(axis=1)
    relative_freq = pivot_df.div(total_counts, axis=0)
    from scipy.stats import entropy
    entropy_values = relative_freq.apply(lambda row: entropy(row), axis=1)
    num_clusters = (pivot_df > 0).sum(axis=1)
    max_entropy = np.log(num_clusters)
    evenness = entropy_values / max_entropy
    simpson = relative_freq.apply(lambda row: 1 - np.sum(row**2), axis=1)
    metrics_df = pd.DataFrame({
        'Entropy': entropy_values,
        'Evenness': evenness,
        'Simpson': simpson
    })
    return metrics_df, freq_df

freq_metrics, freq_df = compute_frequency_metrics(moseq_df)
freq_metrics_reset = freq_metrics.reset_index()

# Mixed-effects models for Shannon Entropy
# Model 1: Control as the reference
model_entropy_control = smf.mixedlm("Entropy ~ New_condition + Experiment", 
                                    data=freq_metrics_reset,
                                    groups=freq_metrics_reset['Animal']).fit(reml=False)
print("🔹 Shannon Entropy Model (Control as reference)")
print(model_entropy_control.summary())
print("AIC:", model_entropy_control.aic, "BIC:", model_entropy_control.bic)

# Model 2: ELS as reference (relevel New_condition)
freq_metrics_reset['New_condition_ELSref'] = pd.Categorical(freq_metrics_reset['New_condition'], 
                                                            categories=['ELS', 'Control', 'ELS_resilient'],
                                                            ordered=True)
model_entropy_els = smf.mixedlm("Entropy ~ C(New_condition_ELSref) + Experiment", 
                                data=freq_metrics_reset,
                                groups=freq_metrics_reset['Animal']).fit(reml=False)
print("🔹 Shannon Entropy Model (ELS as reference)")
print(model_entropy_els.summary())
print("AIC:", model_entropy_els.aic, "BIC:", model_entropy_els.bic)

# Descriptive Stats for Shannon Entropy by New_condition (group by New_condition only)
entropy_stats = freq_metrics_reset.groupby('New_condition').agg(
    N=('Entropy', 'count'),
    mean=('Entropy', 'mean'),
    sd=('Entropy', 'std')
).reset_index()
entropy_stats['sem'] = entropy_stats['sd'] / np.sqrt(entropy_stats['N'])
print("🔹 Entropy Descriptive Stats by New_condition:")
print(entropy_stats)

# Cohen's d for Shannon Entropy comparisons
cd_entropy_control_els = compute_cohen_d(model_entropy_control, 'Control', 'ELS', entropy_stats)
cd_entropy_control_elsres = compute_cohen_d(model_entropy_control, 'Control', 'ELS_resilient', entropy_stats)
cd_entropy_els_elsres = compute_cohen_d(model_entropy_control, 'ELS', 'ELS_resilient', entropy_stats)
print("🔹 Cohen's d (ELS vs Control):", cd_entropy_control_els)
print("🔹 Cohen's d (ELS_resilient vs Control):", cd_entropy_control_elsres)
print("🔹 Cohen's d (ELS vs ELS_resilient):", cd_entropy_els_elsres)

# 📊 Plotting: Shannon Entropy Boxplot
def plot_shannon_entropy_boxplot(df, save_path):
    import numpy as np
    figsize = (3, 4)
    label_color = "#4d4d4d"
    tick_labelsize = 13
    conditions = list(df['New_condition'].unique())
    ordered_conditions = [cond for cond in ['Control', 'ELS', 'ELS_resilient'] if cond in conditions]
    positions = list(range(len(ordered_conditions)))
    fig, ax = plt.subplots(figsize=figsize)
    for i, cond in enumerate(ordered_conditions):
        condition_data = df[df['New_condition'] == cond]['Entropy']
        color = plot_colors.get(cond, label_color)
        ax.boxplot(condition_data, positions=[i], widths=0.5, patch_artist=True,
                   showcaps=True,
                   boxprops={'facecolor': 'none', 'edgecolor': color, 'linewidth': 1},
                   whiskerprops={'color': color, 'linewidth': 1},
                   capprops={'color': color, 'linewidth': 1},
                   medianprops={'color': color, 'linewidth': 1.75},
                   flierprops={'marker': 'o', 'markerfacecolor': color, 'markeredgecolor': color, 'alpha': 0.5},
                   showfliers=True)
        x_jitter = np.random.normal(i, 0.04, size=len(condition_data))
        ax.scatter(x_jitter, condition_data, color=color, alpha=0.75, s=30, edgecolors='none', zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([transform_label(x) for x in ordered_conditions], fontsize=tick_labelsize, color=label_color)
    ax.tick_params(axis="y", labelsize=tick_labelsize, colors=label_color)
    ax.set_ylabel("Shannon entropy index", fontsize=13, color=label_color, labelpad=10)
    ax.set_ylim(1.20, 1.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(label_color)
    ax.spines["bottom"].set_color(label_color)
    plt.tight_layout()
    plt.savefig(save_path, dpi=700, transparent=True, bbox_inches='tight')
    plt.show()

plot_shannon_entropy_boxplot(freq_metrics_reset, r'C:\Users\jsangui\OneDrive - UvA\Others\Downloads\Moseq_final\Clustering\Ethograms\Shannon_entropy_index_resilience.svg')

#%% SECTION 2: Evenness
# Mixed-effects models for Evenness
freq_metrics_reset = freq_metrics.reset_index().reset_index(drop=True)
model_evenness_control = smf.mixedlm("Evenness ~ New_condition + Experiment", 
                                     data=freq_metrics_reset,
                                     groups=freq_metrics_reset['Animal']).fit(reml=False)
print("🔹 Evenness Model (Control as reference)")
print(model_evenness_control.summary())
print("AIC:", model_evenness_control.aic, "BIC:", model_evenness_control.bic)

freq_metrics_reset['New_condition_ELSref'] = pd.Categorical(freq_metrics_reset['New_condition'], 
                                                            categories=['ELS', 'Control', 'ELS_resilient'],
                                                            ordered=True)
model_evenness_els = smf.mixedlm("Evenness ~ C(New_condition_ELSref) + Experiment", 
                                 data=freq_metrics_reset,
                                 groups=freq_metrics_reset['Animal']).fit(reml=False)
print("🔹 Evenness Model (ELS as reference)")
print(model_evenness_els.summary())
print("AIC:", model_evenness_els.aic, "BIC:", model_evenness_els.bic)

evenness_stats = freq_metrics_reset.groupby('New_condition').agg(
    N=('Evenness', 'count'),
    mean=('Evenness', 'mean'),
    sd=('Evenness', 'std')
).reset_index()
evenness_stats['sem'] = evenness_stats['sd'] / np.sqrt(evenness_stats['N'])
print("🔹 Evenness Descriptive Stats by New_condition:")
print(evenness_stats)

cd_even_control_els = compute_cohen_d(model_evenness_control, 'Control', 'ELS', evenness_stats)
cd_even_control_elsres = compute_cohen_d(model_evenness_control, 'Control', 'ELS_resilient', evenness_stats)
cd_even_els_elsres = compute_cohen_d(model_evenness_control, 'ELS', 'ELS_resilient', evenness_stats)
print("🔹 Cohen's d Evenness (ELS vs Control):", cd_even_control_els)
print("🔹 Cohen's d Evenness (ELS_resilient vs Control):", cd_even_control_elsres)
print("🔹 Cohen's d Evenness (ELS vs ELS_resilient):", cd_even_els_elsres)

def plot_evenness_boxplot(df, save_path):
    import numpy as np
    figsize = (3, 4)
    label_color = "#4d4d4d"
    tick_labelsize = 13
    conditions = list(df['New_condition'].unique())
    ordered_conditions = [cond for cond in ['Control', 'ELS', 'ELS_resilient'] if cond in conditions]
    positions = list(range(len(ordered_conditions)))
    fig, ax = plt.subplots(figsize=figsize)
    for i, cond in enumerate(ordered_conditions):
        condition_data = df[df['New_condition'] == cond]['Evenness']
        color = plot_colors.get(cond, label_color)
        ax.boxplot(condition_data, positions=[i], widths=0.5, patch_artist=True,
                   showcaps=True,
                   boxprops={'facecolor': 'none', 'edgecolor': color, 'linewidth': 1},
                   whiskerprops={'color': color, 'linewidth': 1},
                   capprops={'color': color, 'linewidth': 1},
                   medianprops={'color': color, 'linewidth': 1.75},
                   flierprops={'marker': 'o', 'markerfacecolor': color, 'markeredgecolor': color, 'alpha': 0.5},
                   showfliers=True)
        x_jitter = np.random.normal(i, 0.04, size=len(condition_data))
        ax.scatter(x_jitter, condition_data, color=color, alpha=0.75, s=30, edgecolors='none', zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([transform_label(x) for x in ordered_conditions], fontsize=tick_labelsize, color=label_color)
    ax.tick_params(axis="y", labelsize=tick_labelsize, colors=label_color)
    ax.set_ylabel("Evenness index", fontsize=13, color=label_color, labelpad=5)
    ax.set_ylim(0.58, 0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(label_color)
    ax.spines["bottom"].set_color(label_color)
    plt.tight_layout()
    plt.savefig(save_path, dpi=700, transparent=True, bbox_inches='tight')
    plt.show()

plot_evenness_boxplot(freq_metrics_reset, r'C:\Users\jsangui\OneDrive - UvA\Others\Downloads\Moseq_final\Clustering\Ethograms\Eveness_Index_resilience.svg')

#%% SECTION 3: Simpson's Diversity Index
model_simpson_control = smf.mixedlm("Simpson ~ New_condition + Experiment", 
                                    data=freq_metrics_reset,
                                    groups=freq_metrics_reset['Animal']).fit(reml=False)
print("🔹 Simpson Model (Control as reference)")
print(model_simpson_control.summary())
print("AIC:", model_simpson_control.aic, "BIC:", model_simpson_control.bic)

freq_metrics_reset['New_condition_ELSref'] = pd.Categorical(freq_metrics_reset['New_condition'], 
                                                            categories=['ELS', 'Control', 'ELS_resilient'],
                                                            ordered=True)
model_simpson_els = smf.mixedlm("Simpson ~ C(New_condition_ELSref) + Experiment", 
                                data=freq_metrics_reset,
                                groups=freq_metrics_reset['Animal']).fit(reml=False)
print("🔹 Simpson Model (ELS as reference)")
print(model_simpson_els.summary())
print("AIC:", model_simpson_els.aic, "BIC:", model_simpson_els.bic)

simpson_stats = freq_metrics_reset.groupby('New_condition').agg(
    N=('Simpson', 'count'),
    mean=('Simpson', 'mean'),
    sd=('Simpson', 'std')
).reset_index()
simpson_stats['sem'] = simpson_stats['sd'] / np.sqrt(simpson_stats['N'])
print("🔹 Simpson Descriptive Stats by New_condition:")
print(simpson_stats)

cd_sim_control_els = compute_cohen_d(model_simpson_control, 'Control', 'ELS', simpson_stats)
cd_sim_control_elsres = compute_cohen_d(model_simpson_control, 'Control', 'ELS_resilient', simpson_stats)
cd_sim_els_elsres = compute_cohen_d(model_simpson_control, 'ELS', 'ELS_resilient', simpson_stats)
print("🔹 Cohen's d Simpson (ELS vs Control):", cd_sim_control_els)
print("🔹 Cohen's d Simpson (ELS_resilient vs Control):", cd_sim_control_elsres)
print("🔹 Cohen's d Simpson (ELS vs ELS_resilient):", cd_sim_els_elsres)

def plot_simpson_boxplot(df, save_path):
    import numpy as np
    figsize = (3, 4)
    label_color = "#4d4d4d"
    tick_labelsize = 13
    conditions = list(df['New_condition'].unique())
    ordered_conditions = [cond for cond in ['Control', 'ELS', 'ELS_resilient'] if cond in conditions]
    positions = list(range(len(ordered_conditions)))
    fig, ax = plt.subplots(figsize=figsize)
    for i, cond in enumerate(ordered_conditions):
        condition_data = df[df['New_condition'] == cond]['Simpson']
        color = plot_colors.get(cond, label_color)
        ax.boxplot(condition_data, positions=[i], widths=0.5, patch_artist=True,
                   showcaps=True,
                   boxprops={'facecolor': 'none', 'edgecolor': color, 'linewidth': 1},
                   whiskerprops={'color': color, 'linewidth': 1},
                   capprops={'color': color, 'linewidth': 1},
                   medianprops={'color': color, 'linewidth': 1.75},
                   flierprops={'marker': 'o', 'markerfacecolor': color, 'markeredgecolor': color, 'alpha': 0.5},
                   showfliers=True)
        x_jitter = np.random.normal(i, 0.04, size=len(condition_data))
        ax.scatter(x_jitter, condition_data, color=color, alpha=0.75, s=30, edgecolors='none', zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([transform_label(x) for x in ordered_conditions], fontsize=tick_labelsize, color=label_color)
    ax.tick_params(axis="y", labelsize=tick_labelsize, colors=label_color)
    ax.set_ylabel("Simpson index", fontsize=13, color=label_color, labelpad=5)
    ax.set_ylim(0.57, 0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(label_color)
    ax.spines["bottom"].set_color(label_color)
    plt.tight_layout()
    plt.savefig(save_path, dpi=700, transparent=True, bbox_inches='tight')
    plt.show()

plot_simpson_boxplot(freq_metrics_reset, r'C:\Users\jsangui\OneDrive - UvA\Others\Downloads\Moseq_final\Clustering\Ethograms\Simpson_Index_resilience.svg')

#%% SECTION 4: Cumulative Usage Index (CUI)

# 🔍 Data Preprocessing: Filter out rows with missing or empty Cluster names
def compute_freq_usage(freq_df, moseq_df):
    """
    Compute relative usage per cluster from frequency counts.
    
    Parameters:
      freq_df : pandas.DataFrame
          DataFrame with frequency counts. Expected columns include:
          'Animal', 'Experiment', 'Cluster', and 'Count'.
          
      moseq_df : pandas.DataFrame
          DataFrame with metadata. Expected to have at least:
          'Animal' and 'Condition'.
          
    Returns:
      pandas.DataFrame
          A DataFrame with the following added columns:
            - 'Total': Total count per Animal, Condition, and Experiment.
            - 'RelativeUsage': Count divided by Total for each row.
            - 'Condition': Condition merged from moseq_df.
    """
    # Drop any pre-existing Condition column in freq_df to avoid duplicates
    if 'Condition' in freq_df.columns:
        freq_df = freq_df.drop(columns=['Condition'])
    
    # Merge condition information from moseq_df into freq_df
    merged_df = freq_df.merge(
        moseq_df[['Animal', 'Condition']].drop_duplicates(),
        on='Animal',
        how='left'
    )
    
    # Calculate total count per Animal, Condition, and Experiment
    total_counts = merged_df.groupby(['Animal', 'Condition', 'Experiment'])['Count']\
                              .sum().reset_index(name='Total')
    
    # Merge total counts back into the merged DataFrame
    freq_usage = merged_df.merge(total_counts, on=['Animal', 'Condition', 'Experiment'])
    
    # Compute relative usage for each cluster
    freq_usage['RelativeUsage'] = freq_usage['Count'] / freq_usage['Total']
    
    return freq_usage

# Example usage:
freq_usage = compute_freq_usage(freq_df, moseq_df)

# Filter out rows with missing or empty Cluster names
freq_usage = freq_usage.dropna(subset=['Cluster'])
freq_usage = freq_usage[freq_usage['Cluster'].str.strip() != '']

# 🧮 Functions: Compute the refined Cumulative Usage Index (CUI) using compute_cui()
def compute_cui(subdf):
    """
    For a given animal/experiment sub-dataframe (subdf) with relative usage per cluster,
    sort clusters in descending order and compute the cumulative sum.
    Then compute the mean cumulative usage and normalize it as follows:

      baseline = (n + 1) / (2 * n)
      normalized CUI = (mean(cum_usage) - baseline) / (1 - baseline)

    where n is the number of clusters in subdf.
    This normalization scales the index to 0 for a uniform distribution and 1 when one cluster dominates.
    """
    # Sort clusters by descending RelativeUsage
    df_sorted = subdf.sort_values('RelativeUsage', ascending=False).copy()
    # Compute cumulative sum
    cum_usage = df_sorted['RelativeUsage'].cumsum().values
    # Compute mean cumulative usage
    mean_cum = np.mean(cum_usage)
    # Number of clusters
    n = len(cum_usage)
    # Baseline mean cumulative usage under a uniform distribution
    baseline = (n + 1) / (2 * n)
    # Normalize the CUI
    cui = (mean_cum - baseline) / (1 - baseline)
    return cui

# 🔢 Compute the CUI for each Animal/Experiment grouping
# Note: Here we use the original Condition column for merging; you may adjust if you want to use New_condition.
cui_df = freq_usage.groupby(['Animal', 'Experiment', 'Condition']).apply(
    lambda d: compute_cui(d)
).reset_index(name='CUI')

print("Cumulative Usage Index (CUI) per Animal/Experiment:")
print(cui_df.head())

# 📊 Descriptive Stats for CUI
# Now group by New_condition (which includes Control, ELS, and ELS_resilient)
cui_stats = (
    cui_df.merge(moseq_df[['Animal', 'New_condition']].drop_duplicates(), on='Animal', how='left')
    .groupby("New_condition")['CUI']
    .agg(N='count', mean='mean', sd='std')
    .reset_index()
)
cui_stats['sem'] = cui_stats['sd'] / np.sqrt(cui_stats['N'])
print("\nCumulative Usage Index Summary Statistics by New_condition:")
print(cui_stats)

# 📈 Mixed-Effects Model for CUI
# Fit a mixed-effects model with New_condition and Experiment as fixed effects and Animal as a random effect.
# Model 1: Control as reference (default order)
model_cui_control = smf.mixedlm("CUI ~ New_condition + Experiment", cui_df.merge(moseq_df[['Animal', 'New_condition']].drop_duplicates(), on='Animal', how='left'),
                                groups=cui_df["Animal"]).fit(reml=False)
print("\nMixed-Effects Model (Control as reference) Summary:")
print(model_cui_control.summary())

# Model 2: Relevel so that ELS is the intercept
cui_df = cui_df.merge(moseq_df[['Animal', 'New_condition']].drop_duplicates(), on='Animal', how='left')
cui_df['New_condition'] = pd.Categorical(cui_df['New_condition'], categories=["ELS", "Control", "ELS_resilient"], ordered=True)
model_cui_els = smf.mixedlm("CUI ~ New_condition + Experiment", cui_df, groups=cui_df["Animal"]).fit(reml=False)
print("\nMixed-Effects Model (ELS as reference) Summary:")
print(model_cui_els.summary())

# 🔍 Model Fit Statistics: AIC and BIC
print("\nModel Fit Statistics:")
print("AIC:", model_cui_control.aic)
print("BIC:", model_cui_control.bic)

# 📐 Compute Cohen's d for pairwise comparisons among the three conditions
# Using the helper function: compute_cohen_d(model, group1, group2, stats_df)
# Here we pass None for model, as it's not used in the computation.
comparisons = [("Control", "ELS"), ("Control", "ELS_resilient"), ("ELS", "ELS_resilient")]
for group1, group2 in comparisons:
    d = compute_cohen_d(None, group1, group2, cui_stats)
    print(f"\nCohen's d for {group1} vs {group2}: {d:.3f}")

# 📊 Plotting: Create a boxplot with jitter for CUI
def plot_cui_boxplot(df, save_path):
    """
    Plot a boxplot of CUI by New_condition with consistent aesthetics.
    Assumes df has a column 'CUI' and 'New_condition'.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # If df has a multi-index, reset it so that 'New_condition' and 'CUI' are columns
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    figsize = (3, 4)
    label_color = "#4d4d4d"
    tick_labelsize = 13

    # Determine order of conditions (using our new grouping)
    conditions = list(df['New_condition'].unique())
    ordered_conditions = []
    if "Control" in conditions:
        ordered_conditions.append("Control")
    if "ELS" in conditions:
        ordered_conditions.append("ELS")
    if "ELS_resilient" in conditions:
        ordered_conditions.append("ELS_resilient")
    positions = list(range(len(ordered_conditions)))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    for i, cond in enumerate(ordered_conditions):
        condition_data = df[df['New_condition'] == cond]['CUI']
        color = plot_colors.get(cond, label_color)
        ax.boxplot(
            condition_data,
            positions=[i],
            widths=0.5,
            patch_artist=True,
            showcaps=True,
            boxprops={'facecolor': 'none', 'edgecolor': color, 'linewidth': 1},
            whiskerprops={'color': color, 'linewidth': 1},
            capprops={'color': color, 'linewidth': 1},
            medianprops={'color': color, 'linewidth': 1.75},
            flierprops={'marker': 'o', 'markerfacecolor': color, 'markeredgecolor': color, 'alpha': 0.5},
            showfliers=True
        )
        x_jitter = np.random.normal(i, 0.04, size=len(condition_data))
        ax.scatter(x_jitter, condition_data, color=color, alpha=0.75, s=30, edgecolors='none', zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([transform_label(c) for c in ordered_conditions], fontsize=tick_labelsize, color=label_color)
    ax.tick_params(axis="y", labelsize=tick_labelsize, colors=label_color)
    ax.set_ylabel("Cumulative Usage Index", fontsize=13, color=label_color, labelpad=5)
    ax.set_ylim(-0.2, 0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(label_color)
    ax.spines["bottom"].set_color(label_color)
    plt.tight_layout()
    plt.savefig(save_path, dpi=700, transparent=True, bbox_inches='tight')
    plt.show()

# 📁 Save the plot as 'CUI.svg'
plot_cui_boxplot(cui_df, r'C:\Users\jsangui\OneDrive - UvA\Others\Downloads\Moseq_final\Clustering\Ethograms\CUI_resilience.svg')


# 📊 Plotting: area under the curve 

# Define color mapping and label color (updated to include all three groups)
colors = {"Control": "#F3C240", "ELS": "#AD7180", "ELS_resilient": "#90be6d"}
label_color = "#4d4d4d"

# Helper: transform "ELS_resilient" to "ELS resilient" for plot labels
def transform_label(cond):
    return "ELS resilient" if cond == "ELS_resilient" else cond

# 📋 Step 1: Filter the freq_usage DataFrame
print("Unique clusters in freq_usage BEFORE filtering:\n", freq_usage['Cluster'].unique())
freq_usage_filtered = freq_usage.dropna(subset=['Cluster'])  # drop rows with NaN cluster
freq_usage_filtered = freq_usage_filtered[freq_usage_filtered['Cluster'].str.strip() != '']  # drop empty clusters
print("\nUnique clusters AFTER filtering:\n", freq_usage_filtered['Cluster'].unique())

# 📋 Step 2: Group by New_condition & Cluster to get Mean Usage and SEM
# Use New_condition so that all three groups are included.
df_stats = freq_usage_filtered.groupby(['New_condition', 'Cluster'])['RelativeUsage'] \
                .agg(mean='mean', sem='sem').reset_index()
df_stats = df_stats[df_stats['New_condition'].isin(["Control", "ELS", "ELS_resilient"])]
conditions = df_stats['New_condition'].unique()  # get unique conditions

# 🎨 Step 3: Create a figure with 1 large subplot (left) + 3 small subplots (right)
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(14, 8))
# Define a grid with 3 rows and 2 columns; the main plot spans all rows in the left column.
gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[2, 1])
ax_main = fig.add_subplot(gs[:, 0])  # Main subplot (left, spans all rows)
ax_subs = [fig.add_subplot(gs[i, 1]) for i in range(3)]  # Three subplots on the right

# 📈 Step 4: Main subplot (lines with error bars) showing all three conditions
for cond in conditions:
    df_sub = df_stats[df_stats['New_condition'] == cond].copy()
    df_sub.sort_values('mean', ascending=False, inplace=True)  # sort clusters by descending mean usage
    
    cum_usage = df_sub['mean'].cumsum().values  # cumulative mean usage
    # cumulative SEM is approximated assuming independence
    cum_sem = np.sqrt((df_sub['sem']**2).cumsum()).values  
    x = np.arange(len(df_sub))  # x-axis ticks
    
    ax_main.errorbar(x, cum_usage, yerr=cum_sem, marker='o',
                     color=colors.get(cond, label_color), label=transform_label(cond), capsize=3)

ax_main.set_ylim([0.4, 0.9])
ax_main.set_xlabel("")
ax_main.set_ylabel("Cumulative Usage\n", color=label_color, fontsize=14)

# Use the condition with the most clusters for x-tick labels on the main plot
max_cluster_count = 0
max_cluster_cond = None
for cond in conditions:
    count = (df_stats['New_condition'] == cond).sum()
    if count > max_cluster_count:
        max_cluster_count = count
        max_cluster_cond = cond

df_max = df_stats[df_stats['New_condition'] == max_cluster_cond].copy()
df_max.sort_values('mean', ascending=False, inplace=True)
cluster_labels_main = df_max['Cluster'].values
x_ticks_main = np.arange(len(cluster_labels_main))
ax_main.set_xticks(x_ticks_main)
ax_main.set_xticklabels(cluster_labels_main, rotation=0, ha='center', color=label_color, fontsize=12)
ax_main.tick_params(axis='y', labelsize=14, colors=label_color)
for spine in ax_main.spines.values():
    spine.set_visible(True)
    spine.set_color(label_color)
ax_main.tick_params(axis='x', colors=label_color)

import matplotlib.lines as mlines
custom_handles = []
for cond in conditions:
    custom_handles.append(mlines.Line2D([], [], color=colors.get(cond, label_color),
                                          linestyle='-', label=transform_label(cond)))
leg = ax_main.legend(custom_handles, [transform_label(c) for c in conditions], fontsize=12, frameon=False)
for text in leg.get_texts():
    text.set_color("#4d4d4dd4")

# 📉 Step 5: Smaller subplots (one per condition) with filled area and shortened cluster names
# Fix the order of subplots to be: Control, ELS, ELS_resilient
ordered_conditions = ["Control", "ELS", "ELS_resilient"]
for i, cond in enumerate(ordered_conditions):
    df_sub = df_stats[df_stats['New_condition'] == cond].copy()
    df_sub.sort_values('mean', ascending=False, inplace=True)
    cum_usage = df_sub['mean'].cumsum().values
    x_vals = np.arange(len(df_sub))
    # Shorten cluster names to the first 3 characters
    clusters_short = df_sub['Cluster'].apply(lambda s: s[:3]).values
    
    ax = ax_subs[i]
    ax.fill_between(x_vals, cum_usage, alpha=0.2, color=colors.get(cond, label_color))
    ax.plot(x_vals, cum_usage, marker='o', color=colors.get(cond, label_color))
    ax.set_title(transform_label(cond), color=label_color)
    ax.set_ylim([0, 1])
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative Usage\n", color=label_color)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(clusters_short, rotation=0, ha='center', color=label_color)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(label_color)
    ax.spines['left'].set_color(label_color)
    ax.tick_params(axis='x', colors=label_color)
    ax.tick_params(axis='y', colors=label_color)

plt.tight_layout()
plt.savefig(r"C:\Users\jsangui\OneDrive - UvA\Others\Downloads\Moseq_final\Clustering\Ethograms\CUI_area_under_curve_resilience.svg",
            dpi=700, transparent=True, bbox_inches='tight')
plt.show()





#%% SECTION 5: Mean Bout Duration – Overall Analysis
def compute_mean_bout_duration(subdf):
    subdf = subdf.sort_values('Time_bin')
    subdf['Cluster_change'] = (subdf['Cluster'] != subdf['Cluster'].shift(1)).astype(int)
    subdf['Bout'] = subdf['Cluster_change'].cumsum()
    bout_durations = subdf.groupby('Bout').size() * 0.25
    return bout_durations.mean()

bout_duration_overall = moseq_df.groupby(['Animal', 'Experiment']).apply(compute_mean_bout_duration).reset_index(name='MeanBoutDuration')
bout_duration_overall = bout_duration_overall.merge(
    moseq_df[['Animal', 'New_condition', 'Experiment']].drop_duplicates(), 
    on=['Animal', 'Experiment'], 
    how='left'
)

model_bout_control = smf.mixedlm("MeanBoutDuration ~ New_condition + Experiment", bout_duration_overall, groups=bout_duration_overall["Animal"]).fit(reml=False)
print("🔹 Overall Bout Duration Model (Control as reference)")
print(model_bout_control.summary())
print("AIC:", model_bout_control.aic, "BIC:", model_bout_control.bic)

bout_duration_overall['New_condition_ELSref'] = pd.Categorical(bout_duration_overall['New_condition'], 
                                                               categories=['ELS', 'Control', 'ELS_resilient'],
                                                               ordered=True)
model_bout_els = smf.mixedlm("MeanBoutDuration ~ C(New_condition_ELSref) + Experiment", bout_duration_overall, groups=bout_duration_overall["Animal"]).fit(reml=False)
print("🔹 Overall Bout Duration Model (ELS as reference)")
print(model_bout_els.summary())
print("AIC:", model_bout_els.aic, "BIC:", model_bout_els.bic)

bout_stats = bout_duration_overall.groupby("New_condition").agg(
    N=('MeanBoutDuration', 'count'),
    mean=('MeanBoutDuration', 'mean'),
    sd=('MeanBoutDuration', 'std')
).reset_index()
bout_stats['sem'] = bout_stats['sd'] / np.sqrt(bout_stats['N'])
print("🔹 Overall Bout Duration Descriptive Stats:")
print(bout_stats)

cd_bout_control_els = compute_cohen_d(model_bout_control, 'Control', 'ELS', bout_stats)
cd_bout_control_elsres = compute_cohen_d(model_bout_control, 'Control', 'ELS_resilient', bout_stats)
cd_bout_els_elsres = compute_cohen_d(model_bout_control, 'ELS', 'ELS_resilient', bout_stats)
print("🔹 Cohen's d Bout Duration (ELS vs Control):", cd_bout_control_els)
print("🔹 Cohen's d Bout Duration (ELS_resilient vs Control):", cd_bout_control_elsres)
print("🔹 Cohen's d Bout Duration (ELS vs ELS_resilient):", cd_bout_els_elsres)

def plot_bout_duration_overall_boxplot(df, save_path):
    import numpy as np
    figsize = (3, 4)
    label_color = "#4d4d4d"
    tick_labelsize = 13
    conditions = list(df['New_condition'].unique())
    ordered_conditions = [cond for cond in ['Control', 'ELS', 'ELS_resilient'] if cond in conditions]
    positions = list(range(len(ordered_conditions)))
    fig, ax = plt.subplots(figsize=figsize)
    for i, cond in enumerate(ordered_conditions):
        condition_data = df[df['New_condition'] == cond]['MeanBoutDuration']
        color = plot_colors.get(cond, label_color)
        ax.boxplot(condition_data, positions=[i], widths=0.5, patch_artist=True,
                   showcaps=True,
                   boxprops={'facecolor': 'none', 'edgecolor': color, 'linewidth': 1},
                   whiskerprops={'color': color, 'linewidth': 1},
                   capprops={'color': color, 'linewidth': 1},
                   medianprops={'color': color, 'linewidth': 1.75},
                   flierprops={'marker': 'o', 'markerfacecolor': color, 'markeredgecolor': color, 'alpha': 0.5},
                   showfliers=True)
        x_jitter = np.random.normal(i, 0.04, size=len(condition_data))
        ax.scatter(x_jitter, condition_data, color=color, alpha=0.75, s=25, edgecolors='none', zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([transform_label(x) for x in ordered_conditions], fontsize=tick_labelsize, color=label_color)
    ax.tick_params(axis="y", labelsize=tick_labelsize, colors=label_color)
    ax.set_ylabel("Mean Bout Duration (s)", fontsize=13, color=label_color, labelpad=10)
    ax.set_ylim(0.6, 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(label_color)
    ax.spines["bottom"].set_color(label_color)
    plt.tight_layout()
    plt.savefig(save_path, dpi=700, transparent=True, bbox_inches='tight')
    plt.show()

plot_bout_duration_overall_boxplot(bout_duration_overall, r'C:\Users\jsangui\OneDrive - UvA\Others\Downloads\Moseq_final\Clustering\Ethograms\MeanBoutDuration_Overall.svg')

#%% SECTION 6: Mean Bout Duration – Per Cluster Analysis
def compute_cluster_bout_duration(subdf):
    subdf = subdf.sort_values('Time_bin')
    subdf['Cluster_change'] = (subdf['Cluster'] != subdf['Cluster'].shift(1)).astype(int)
    subdf['Bout'] = subdf['Cluster_change'].cumsum()
    bout_durations = subdf.groupby(['Bout', 'Cluster']).size() * 0.25
    return bout_durations.groupby(level=1).mean()

cluster_bout_duration = moseq_df.groupby(['Animal', 'Experiment']).apply(compute_cluster_bout_duration).reset_index()
cluster_bout_duration.columns = ['Animal', 'Experiment', 'Cluster', 'MeanBoutDuration']
cluster_bout_duration = cluster_bout_duration.dropna(subset=['MeanBoutDuration']).reset_index(drop=True)
cluster_bout_duration = cluster_bout_duration.merge(
    moseq_df[['Animal', 'New_condition', 'Experiment']].drop_duplicates(), 
    on=['Animal', 'Experiment'], 
    how='left'
)

ordered_clusters = ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]

print("🔹 Per Cluster Mixed-Effects Models and Descriptive Stats:")
for cluster in ordered_clusters:
    df_cluster = cluster_bout_duration[cluster_bout_duration['Cluster'] == cluster]
    if len(df_cluster) > 0:
        stats = df_cluster.groupby("New_condition").agg(
            N=('MeanBoutDuration', 'count'),
            mean=('MeanBoutDuration', 'mean'),
            sd=('MeanBoutDuration', 'std')
        ).reset_index()
        stats['sem'] = stats['sd'] / np.sqrt(stats['N'])
        print(f"Descriptive Stats for {cluster}:")
        print(stats)
        
        model_cluster_control = smf.mixedlm("MeanBoutDuration ~ New_condition + Experiment", 
                                            df_cluster, 
                                            groups=df_cluster["Animal"]).fit(reml=False)
        print(f"Mixed-Effects Model for {cluster} (Control as reference):")
        print(model_cluster_control.summary())
        print("AIC:", model_cluster_control.aic, "BIC:", model_cluster_control.bic)
        
        df_cluster['New_condition_ELSref'] = pd.Categorical(df_cluster['New_condition'],
                                                            categories=['ELS', 'Control', 'ELS_resilient'],
                                                            ordered=True)
        model_cluster_els = smf.mixedlm("MeanBoutDuration ~ C(New_condition_ELSref) + Experiment", 
                                        df_cluster, 
                                        groups=df_cluster["Animal"]).fit(reml=False)
        print(f"Mixed-Effects Model for {cluster} (ELS as reference):")
        print(model_cluster_els.summary())
        
        # coef = model_cluster_control.params.get('New_condition[T.ELS]', np.nan)
        # resid_sd = np.sqrt(model_cluster_control.scale)
        # cohen_d_els_vs_ctrl = coef_els / resid_sd
        # cohen_d_elsres_vs_ctrl = coef_els_res / resid_sd
        # cohen_d_els_vs_elsres = (coef_els - coef_els_res) / resid_sd

        # print("Cohen's d (ELS vs Control):", cohen_d)
        # print("-" * 40)
    else:
        print(f"No data for cluster: {cluster}")
        print("-" * 40)

def plot_bout_duration(cluster_df, save_path):
    import matplotlib.gridspec as gridspec
    cluster_df = cluster_df[cluster_df['Cluster'].astype(str).str.strip() != ""]
    ordered_clusters = ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]
    y_max_mapping = {"Freezing": 2, "Sniffing": 3.5, "Grooming": 0.6, "Turn": 2.5, "Locomotion": 0.8, "Climbing": 2, "Jump": 1.2}
    
    facet_width, facet_height = 2.75, 4
    fig_width = 4 * facet_width
    fig_height = 2 * facet_height
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(2, 4)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(4)]
    
    label_color = "#4d4d4d"
    tick_labelsize = 13

    for ax, cluster in zip(axes, ordered_clusters):
        data = cluster_df[cluster_df['Cluster'] == cluster]
        conditions = list(data['New_condition'].unique())
        ordered_conditions = [cond for cond in ['Control', 'ELS', 'ELS_resilient'] if cond in conditions]
        positions = list(range(len(ordered_conditions)))
        for j, cond in enumerate(ordered_conditions):
            condition_data = data[data['New_condition'] == cond]['MeanBoutDuration']
            color = plot_colors.get(cond, label_color)
            ax.boxplot(condition_data, positions=[j], widths=0.5, patch_artist=True,
                       showcaps=True,
                       boxprops={'facecolor': 'none', 'edgecolor': color, 'linewidth': 1},
                       whiskerprops={'color': color, 'linewidth': 1},
                       capprops={'color': color, 'linewidth': 1},
                       medianprops={'color': color, 'linewidth': 1.75},
                       flierprops={'marker': 'o', 'markerfacecolor': color, 'markeredgecolor': color, 'alpha': 0.5},
                       showfliers=True)
            x_jitter = np.random.normal(j, 0.04, size=len(condition_data))
            ax.scatter(x_jitter, condition_data, color=color, alpha=0.75, s=25, edgecolors='none', zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([transform_label(x) for x in ordered_conditions], fontsize=tick_labelsize, color=label_color)
        ax.tick_params(axis="y", labelsize=tick_labelsize, colors=label_color)
        ax.set_ylabel("Mean Bout Duration (s)", fontsize=13, color=label_color, labelpad=10)
        ax.set_title(cluster, fontsize=14, color=label_color, pad=20)
        if cluster in y_max_mapping:
            ax.set_ylim(0, y_max_mapping[cluster])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(label_color)
        ax.spines["bottom"].set_color(label_color)
    
    if len(axes) > len(ordered_clusters):
        for i in range(len(ordered_clusters), len(axes)):
            axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=700, transparent=True, bbox_inches='tight')
    plt.show()

plot_bout_duration(cluster_bout_duration, r'C:\Users\jsangui\OneDrive - UvA\Others\Downloads\Moseq_final\Clustering\Ethograms\MeanBoutDuration_PerCluster.svg')

#%%
# Ensure that bout_duration_overall and cluster_bout_duration are defined
# For overall bout duration:
# bout_duration_overall is assumed to have columns: Animal, Experiment, MeanBoutDuration, New_condition

# For per-cluster bout duration:
# cluster_bout_duration is assumed to have columns: Animal, Experiment, Cluster, MeanBoutDuration, New_condition

# Create a copy and rename New_condition to Condition in both DataFrames
bout_duration_overall = bout_duration_overall.copy()
bout_duration_overall['Cluster'] = "Overall"  # assign overall bout durations a cluster label 'Overall'
bout_duration_overall = bout_duration_overall.rename(columns={'New_condition': 'Condition'})

cluster_bout_duration = cluster_bout_duration.copy()
cluster_bout_duration = cluster_bout_duration.rename(columns={'New_condition': 'Condition'})

# Combine the two DataFrames into one
combined_bout_duration = pd.concat(
    [bout_duration_overall[['Animal', 'Experiment', 'Cluster', 'MeanBoutDuration', 'Condition']],
     cluster_bout_duration[['Animal', 'Experiment', 'Cluster', 'MeanBoutDuration', 'Condition']]],
    ignore_index=True
)
#%%
def plot_bout_duration(cluster_df, save_path):
    """
    Plot multi-panel boxplots for Mean Bout Duration per cluster,
    including an "Overall" panel and a panel for each cluster.
    This function accommodates three groups:
      - Control
      - ELS
      - ELS_resilient (displayed as two-line label: "ELS" then "resilient")

    The function:
      - Removes rows with empty cluster names.
      - Uses a predefined order for clusters.
      - Creates an 8-panel (2x4) layout with increased spacing.
      - For each cluster, plots a boxplot with jitter for each Condition.

    Parameters:
      cluster_df : pandas.DataFrame
          DataFrame with columns 'Cluster', 'MeanBoutDuration', and 'Condition'.
      save_path : str
          File path to save the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec

    # Helper: transform condition label for x-axis
    def transform_label(cond):
        if cond == "ELS_resilient":
            return "ELS\nresilient"
        else:
            return cond

    # Remove rows with empty cluster names
    cluster_df = cluster_df[cluster_df['Cluster'].astype(str).str.strip() != ""]

    ordered_clusters = ["Freezing", "Sniffing", "Grooming", "Turn", 
                        "Locomotion", "Climbing", "Jump", "Overall"]
    y_max_mapping = {"Freezing": 2, "Sniffing": 4, "Grooming": 0.6, "Turn": 2.5, 
                     "Locomotion": 0.8, "Climbing": 2, "Jump": 1.2, "Overall": 3}

    facet_width, facet_height = 3.6, 4
    fig_width = 4 * facet_width
    fig_height = 2 * facet_height
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Increase spacing between subplots
    gs = gridspec.GridSpec(2, 4, wspace=0.9, hspace=0.5)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(4)]

    # Updated color mapping including resilience
    colors = {"Control": "#F3C240", "ELS": "#AD7180", "ELS_resilient": "#90be6d"}
    label_color = "#4d4d4d"
    tick_labelsize = 13

    # Plot for each cluster (8 panels)
    for ax, cluster in zip(axes, ordered_clusters):
        data = cluster_df[cluster_df['Cluster'] == cluster]
        conditions = list(data['Condition'].unique())
        # Order conditions: Control, ELS, ELS_resilient (if present) then any others
        ordered_conditions = []
        if "Control" in conditions:
            ordered_conditions.append("Control")
        if "ELS" in conditions:
            ordered_conditions.append("ELS")
        if "ELS_resilient" in conditions:
            ordered_conditions.append("ELS_resilient")
        for cond in conditions:
            if cond not in ordered_conditions:
                ordered_conditions.append(cond)
        positions = list(range(len(ordered_conditions)))
        for j, cond in enumerate(ordered_conditions):
            condition_data = data[data['Condition'] == cond]['MeanBoutDuration']
            color = colors.get(cond, label_color)
            ax.boxplot(
                condition_data,
                positions=[j],
                widths=0.5,
                patch_artist=True,
                showcaps=True,
                boxprops={'facecolor': 'none', 'edgecolor': color, 'linewidth': 1},
                whiskerprops={'color': color, 'linewidth': 1},
                capprops={'color': color, 'linewidth': 1},
                medianprops={'color': color, 'linewidth': 1.75},
                flierprops={'marker': 'o', 'markerfacecolor': color, 'markeredgecolor': color, 'alpha': 0.5},
                showfliers=True
            )
            x_jitter = np.random.normal(j, 0.04, size=len(condition_data))
            ax.scatter(x_jitter, condition_data, color=color, alpha=0.75, s=25, edgecolors='none', zorder=3)
        # Transform each label if necessary
        transformed_labels = [transform_label(cond) for cond in ordered_conditions]
        ax.set_xticks(positions)
        ax.set_xticklabels(transformed_labels, fontsize=tick_labelsize, color=label_color)
        ax.tick_params(axis="y", labelsize=tick_labelsize, colors=label_color)
        ax.set_ylabel("Mean Bout Duration (s)", fontsize=13, color=label_color, labelpad=10)
        ax.set_title(cluster, fontsize=14, color=label_color, pad=20)
        if cluster in y_max_mapping:
            ax.set_ylim(0, y_max_mapping[cluster])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(label_color)
        ax.spines["bottom"].set_color(label_color)

    # Turn off any extra subplot(s) if present
    if len(axes) > len(ordered_clusters):
        for i in range(len(ordered_clusters), len(axes)):
            axes[i].axis("off")

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=700, transparent=True, bbox_inches='tight')
    plt.show()


plot_bout_duration(combined_bout_duration, r'C:\Users\jsangui\OneDrive - UvA\Others\Downloads\Moseq_final\Clustering\Ethograms\MeanBoutDuration_PerCluster_and_overall_resilient.svg')
