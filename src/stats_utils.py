"""Statistical testing utilities for coping-dynamics-sequencing."""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Optional, Dict, Any


def ttest_ind_groups(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    group1: str,
    group2: str,
    equal_var: bool = False,
) -> Dict[str, Any]:
    """
    Perform independent samples t-test between two groups.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data
    value_col : str
        Column name for the values to compare
    group_col : str
        Column name for group labels
    group1, group2 : str
        Names of the two groups to compare
    equal_var : bool
        If False (default), perform Welch's t-test
        
    Returns
    -------
    dict with keys: statistic, pvalue, effect_size (Cohen's d), 
                    n1, n2, mean1, mean2, std1, std2
    """
    g1 = data.loc[data[group_col] == group1, value_col].dropna()
    g2 = data.loc[data[group_col] == group2, value_col].dropna()
    
    stat, pval = stats.ttest_ind(g1, g2, equal_var=equal_var)
    
    # Cohen's d
    pooled_std = np.sqrt(((len(g1) - 1) * g1.std()**2 + (len(g2) - 1) * g2.std()**2) / 
                         (len(g1) + len(g2) - 2))
    cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else np.nan
    
    return {
        "statistic": stat,
        "pvalue": pval,
        "effect_size": cohens_d,
        "n1": len(g1),
        "n2": len(g2),
        "mean1": g1.mean(),
        "mean2": g2.mean(),
        "std1": g1.std(),
        "std2": g2.std(),
        "group1": group1,
        "group2": group2,
    }


def mannwhitneyu_groups(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    group1: str,
    group2: str,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """
    Perform Mann-Whitney U test between two groups (non-parametric).
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data
    value_col : str
        Column name for the values to compare
    group_col : str
        Column name for group labels
    group1, group2 : str
        Names of the two groups to compare
    alternative : str
        'two-sided', 'less', or 'greater'
        
    Returns
    -------
    dict with keys: statistic, pvalue, effect_size (rank-biserial r),
                    n1, n2, median1, median2
    """
    g1 = data.loc[data[group_col] == group1, value_col].dropna()
    g2 = data.loc[data[group_col] == group2, value_col].dropna()
    
    stat, pval = stats.mannwhitneyu(g1, g2, alternative=alternative)
    
    # Rank-biserial correlation as effect size
    n1, n2 = len(g1), len(g2)
    r = 1 - (2 * stat) / (n1 * n2)
    
    return {
        "statistic": stat,
        "pvalue": pval,
        "effect_size": r,
        "n1": n1,
        "n2": n2,
        "median1": g1.median(),
        "median2": g2.median(),
        "group1": group1,
        "group2": group2,
    }


def one_way_anova(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    groups: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Perform one-way ANOVA across multiple groups.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data
    value_col : str
        Column name for the values to compare
    group_col : str
        Column name for group labels
    groups : list of str, optional
        Specific groups to include; if None, use all unique groups
        
    Returns
    -------
    dict with keys: statistic, pvalue, effect_size (eta-squared),
                    group_means, group_stds, group_ns
    """
    if groups is None:
        groups = data[group_col].unique().tolist()
    
    group_data = [data.loc[data[group_col] == g, value_col].dropna() for g in groups]
    
    stat, pval = stats.f_oneway(*group_data)
    
    # Eta-squared
    all_vals = np.concatenate(group_data)
    grand_mean = all_vals.mean()
    ss_between = sum(len(gd) * (gd.mean() - grand_mean)**2 for gd in group_data)
    ss_total = sum((all_vals - grand_mean)**2)
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
    
    return {
        "statistic": stat,
        "pvalue": pval,
        "effect_size": eta_sq,
        "groups": groups,
        "group_means": {g: gd.mean() for g, gd in zip(groups, group_data)},
        "group_stds": {g: gd.std() for g, gd in zip(groups, group_data)},
        "group_ns": {g: len(gd) for g, gd in zip(groups, group_data)},
    }


def kruskal_wallis(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    groups: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Perform Kruskal-Wallis H-test (non-parametric ANOVA).
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data
    value_col : str
        Column name for the values to compare
    group_col : str
        Column name for group labels
    groups : list of str, optional
        Specific groups to include; if None, use all unique groups
        
    Returns
    -------
    dict with keys: statistic, pvalue, effect_size (epsilon-squared),
                    group_medians, group_ns
    """
    if groups is None:
        groups = data[group_col].unique().tolist()
    
    group_data = [data.loc[data[group_col] == g, value_col].dropna() for g in groups]
    
    stat, pval = stats.kruskal(*group_data)
    
    # Epsilon-squared as effect size
    n = sum(len(gd) for gd in group_data)
    epsilon_sq = stat / (n - 1) if n > 1 else np.nan
    
    return {
        "statistic": stat,
        "pvalue": pval,
        "effect_size": epsilon_sq,
        "groups": groups,
        "group_medians": {g: gd.median() for g, gd in zip(groups, group_data)},
        "group_ns": {g: len(gd) for g, gd in zip(groups, group_data)},
    }


def pairwise_posthoc(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    groups: Optional[List[str]] = None,
    method: str = "bonferroni",
    parametric: bool = True,
) -> pd.DataFrame:
    """
    Perform pairwise post-hoc comparisons with multiple testing correction.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data
    value_col : str
        Column name for the values to compare
    group_col : str
        Column name for group labels
    groups : list of str, optional
        Specific groups to include; if None, use all unique groups
    method : str
        Correction method: 'bonferroni', 'holm', 'fdr_bh' (Benjamini-Hochberg)
    parametric : bool
        If True, use t-test; if False, use Mann-Whitney U
        
    Returns
    -------
    pd.DataFrame with columns: group1, group2, statistic, pvalue, pvalue_corrected, significant
    """
    from itertools import combinations
    from statsmodels.stats.multitest import multipletests
    
    if groups is None:
        groups = data[group_col].unique().tolist()
    
    results = []
    for g1, g2 in combinations(groups, 2):
        if parametric:
            res = ttest_ind_groups(data, value_col, group_col, g1, g2)
        else:
            res = mannwhitneyu_groups(data, value_col, group_col, g1, g2)
        results.append({
            "group1": g1,
            "group2": g2,
            "statistic": res["statistic"],
            "pvalue": res["pvalue"],
            "effect_size": res["effect_size"],
        })
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        _, pvals_corrected, _, _ = multipletests(df["pvalue"], method=method)
        df["pvalue_corrected"] = pvals_corrected
        df["significant"] = df["pvalue_corrected"] < 0.05
    
    return df


def format_pvalue(p: float, threshold: float = 0.001) -> str:
    """Format p-value for display."""
    if p < threshold:
        return f"p < {threshold}"
    return f"p = {p:.3f}"


def significance_stars(p: float) -> str:
    """Return significance stars based on p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def report_test(result: Dict[str, Any], test_name: str = "test") -> str:
    """
    Generate a formatted report string for a statistical test result.
    
    Parameters
    ----------
    result : dict
        Output from one of the test functions
    test_name : str
        Name of the test for the report
        
    Returns
    -------
    str : Formatted report string
    """
    p = result["pvalue"]
    stat = result["statistic"]
    effect = result.get("effect_size", np.nan)
    
    report = f"{test_name}: statistic={stat:.3f}, {format_pvalue(p)}"
    if not np.isnan(effect):
        report += f", effect size={effect:.3f}"
    report += f" {significance_stars(p)}"
    
    return report
