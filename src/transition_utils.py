"""
Transition matrix analysis utilities.

Functions for computing transition matrices, stabilized transition matrices,
BFL (Behavioral Flow Likeness) scores, and bootstrap permutation tests.

These functions are adapted from the reference implementation in:
  keypoint_moseq_project/code/resilience_analysis_v6.py
"""

import numpy as np
import pandas as pd
import random
from scipy.special import erf


def compute_transition_matrix(seq):
    """
    Remove consecutive duplicates from a sequence and compute its transition matrix.
    
    This approach focuses on state *transitions* rather than raw frame counts,
    removing self-loops from the analysis.
    
    Parameters
    ----------
    seq : array-like
        Sequence of behavioral state codes (e.g., syllable labels).
    
    Returns
    -------
    pd.DataFrame
        Transition matrix with rows = from-states and columns = to-states.
        Values are raw counts (not normalized).
    """
    if len(seq) == 0:
        return pd.DataFrame()
    
    # Remove consecutive duplicates
    filtered = [seq[0]]
    for s in seq[1:]:
        if s != filtered[-1]:
            filtered.append(s)
    
    # Count transitions
    pairs = list(zip(filtered[:-1], filtered[1:]))
    states = sorted(set(filtered))
    TM = pd.DataFrame(0, index=states, columns=states, dtype=float)
    for a, b in pairs:
        TM.loc[a, b] += 1
    
    return TM


def compute_stabilized_transition_matrices(trans_mats, group_assignments, control_value="Control"):
    """
    Compute stabilized transition matrices by subtracting the control group average.
    
    This normalization highlights deviations from typical control behavior,
    making group differences more apparent.
    
    Parameters
    ----------
    trans_mats : dict
        Dictionary mapping recording names to transition matrices (DataFrames).
    group_assignments : dict
        Dictionary mapping recording names to group labels.
    control_value : str, default "Control"
        Label for the control group.
    
    Returns
    -------
    dict
        Dictionary mapping recording names to stabilized transition matrices.
    """
    # Get control recordings
    control_keys = [rec for rec, grp in group_assignments.items() 
                    if grp.lower() == control_value.lower() and rec in trans_mats]
    
    if not control_keys:
        raise ValueError("No recordings found for the control group.")
    
    # Compute average control transition matrix
    avg_control = None
    for rec in control_keys:
        if avg_control is None:
            avg_control = trans_mats[rec].copy()
        else:
            avg_control = avg_control.add(trans_mats[rec], fill_value=0)
    avg_control /= len(control_keys)
    
    # Subtract control average from each recording
    stabilized = {}
    for rec, tm in trans_mats.items():
        tm_aligned = tm.reindex(index=avg_control.index, columns=avg_control.columns, fill_value=0)
        stabilized[rec] = tm_aligned - avg_control
    
    return stabilized


def relabel_matrix(tm, mapping):
    """
    Relabel a transition matrix's rows and columns using the provided mapping.
    
    Parameters
    ----------
    tm : pd.DataFrame
        Transition matrix to relabel.
    mapping : dict
        Dictionary mapping old labels to new labels.
    
    Returns
    -------
    pd.DataFrame
        Relabeled transition matrix.
    """
    tm = tm.copy()
    tm.rename(index=mapping, columns=mapping, inplace=True)
    return tm


def compute_group_mean(mats):
    """Compute the element-wise mean of a list of matrices."""
    return np.mean(np.stack(mats, axis=0), axis=0)


def compute_group_median(mats):
    """Compute the element-wise median of a list of matrices."""
    return np.median(np.stack(mats, axis=0), axis=0)


def manhattan_distance(mat1, mat2):
    """
    Compute the Manhattan distance (L1 norm) between two matrices.
    
    Parameters
    ----------
    mat1, mat2 : array-like or pd.DataFrame
        Matrices to compare.
    
    Returns
    -------
    float
        Manhattan distance between the matrices.
    """
    arr1 = mat1.values if hasattr(mat1, 'values') else np.array(mat1)
    arr2 = mat2.values if hasattr(mat2, 'values') else np.array(mat2)
    return np.sum(np.abs(arr1 - arr2))


def bootstrap_intergroup_distance(transition_mats, group_assignments, group1, group2, 
                                   n_bootstraps=1000, seed=123):
    """
    Bootstrap permutation test for intergroup transition matrix distance.
    
    Computes the true Manhattan distance between group mean transition matrices,
    then generates a null distribution by randomly shuffling group labels.
    
    Parameters
    ----------
    transition_mats : dict
        Dictionary mapping recording names to transition matrices (as numpy arrays or DataFrames).
    group_assignments : dict
        Dictionary mapping recording names to group labels.
    group1, group2 : str
        Names of the two groups to compare.
    n_bootstraps : int, default 1000
        Number of bootstrap permutations.
    seed : int, default 123
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Dictionary containing:
        - true_distance: Manhattan distance between actual group means
        - bootstrap_mean: Mean of null distribution
        - bootstrap_std: Std of null distribution
        - percentile: Percentile of true distance in null distribution
        - sigma: Z-score of true distance
        - p_value: One-tailed p-value
        - bootstrap_distances: Array of null distribution distances
    """
    # Get recordings for each group
    recs = [rec for rec in transition_mats 
            if group_assignments.get(rec, None) in (group1, group2)]
    recs_g1 = [rec for rec in recs if group_assignments[rec] == group1]
    recs_g2 = [rec for rec in recs if group_assignments[rec] == group2]
    
    if not recs_g1 or not recs_g2:
        raise ValueError("One of the groups has no recordings.")
    
    # Compute true distance
    mean1 = compute_group_mean([transition_mats[rec].values if hasattr(transition_mats[rec], 'values') 
                                else transition_mats[rec] for rec in recs_g1])
    mean2 = compute_group_mean([transition_mats[rec].values if hasattr(transition_mats[rec], 'values') 
                                else transition_mats[rec] for rec in recs_g2])
    true_distance = manhattan_distance(mean1, mean2)
    
    # Bootstrap permutation test
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
        
        mean_boot1 = compute_group_mean([transition_mats[rec].values if hasattr(transition_mats[rec], 'values') 
                                         else transition_mats[rec] for rec in boot_g1])
        mean_boot2 = compute_group_mean([transition_mats[rec].values if hasattr(transition_mats[rec], 'values') 
                                         else transition_mats[rec] for rec in boot_g2])
        bootstrap_dists.append(manhattan_distance(mean_boot1, mean_boot2))
    
    bootstrap_dists = np.array(bootstrap_dists)
    boot_mean = np.mean(bootstrap_dists)
    boot_std = np.std(bootstrap_dists)
    percentile = (np.sum(bootstrap_dists < true_distance) / (len(bootstrap_dists) + 1)) * 100
    sigma = (true_distance - boot_mean) / boot_std if boot_std > 0 else np.nan
    p_value = 0.5 * (1 - erf(sigma / np.sqrt(2))) if not np.isnan(sigma) else np.nan
    
    return {
        'true_distance': true_distance,
        'bootstrap_mean': boot_mean,
        'bootstrap_std': boot_std,
        'percentile': percentile,
        'sigma': sigma,
        'p_value': p_value,
        'bootstrap_distances': bootstrap_dists
    }


def compute_bfl_scores(transition_mats, group_assignments, group1="Control", group2="ELS"):
    """
    Compute Behavioral Flow Likeness (BFL) scores for each recording.
    
    BFL score measures how similar a recording's transition pattern is to each group's
    typical pattern. Positive scores indicate similarity to group2 (ELS),
    negative scores indicate similarity to group1 (Control).
    
    Parameters
    ----------
    transition_mats : dict
        Dictionary mapping recording names to transition matrices.
    group_assignments : dict
        Dictionary mapping recording names to group labels.
    group1 : str, default "Control"
        Name of the first group (typically Control).
    group2 : str, default "ELS"
        Name of the second group (typically ELS).
    
    Returns
    -------
    dict
        Dictionary mapping recording names to BFL scores.
        BFL = log(d_group1 / d_group2) where d is Manhattan distance to group median.
    """
    # Get recordings for each group
    recs_g1 = [rec for rec in transition_mats if group_assignments.get(rec, None) == group1]
    recs_g2 = [rec for rec in transition_mats if group_assignments.get(rec, None) == group2]
    
    if not recs_g1 or not recs_g2:
        raise ValueError("One of the groups has no recordings.")
    
    # Compute group medians
    median1 = compute_group_median([transition_mats[rec].values if hasattr(transition_mats[rec], 'values') 
                                    else transition_mats[rec] for rec in recs_g1])
    median2 = compute_group_median([transition_mats[rec].values if hasattr(transition_mats[rec], 'values') 
                                    else transition_mats[rec] for rec in recs_g2])
    
    # Compute BFL score for each recording
    bfl_scores = {}
    for rec, mat in transition_mats.items():
        if group_assignments.get(rec, None) not in (group1, group2):
            continue
        arr = mat.values if hasattr(mat, 'values') else mat
        dA = manhattan_distance(arr, median1)
        dB = manhattan_distance(arr, median2)
        eps = 1e-10
        bfl = np.log((dA + eps) / (dB + eps))
        bfl_scores[rec] = bfl
    
    return bfl_scores


def compute_mean_transition_matrices_by_group(trans_mats, group_assignments):
    """
    Compute mean transition matrices for each group.
    
    Parameters
    ----------
    trans_mats : dict
        Dictionary mapping recording names to transition matrices.
    group_assignments : dict
        Dictionary mapping recording names to group labels.
    
    Returns
    -------
    dict
        Dictionary mapping group names to mean transition matrices.
    """
    group_sums = {}
    group_counts = {}
    
    for rec, tm in trans_mats.items():
        grp = group_assignments.get(rec)
        if grp is None:
            continue
        if grp not in group_sums:
            group_sums[grp] = tm.copy()
            group_counts[grp] = 1
        else:
            group_sums[grp] = group_sums[grp].add(tm, fill_value=0)
            group_counts[grp] += 1
    
    group_means = {}
    for grp in group_sums:
        group_means[grp] = group_sums[grp] / group_counts[grp]
    
    return group_means
