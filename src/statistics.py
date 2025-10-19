"""Statistical analysis functions."""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test, logrank_test
from scipy import stats


def compute_multivariate_logrank(df, time_col, event_col, group_col):
    """
    Compute multivariate log-rank test.
    
    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    event_col : str
    group_col : str
    
    Returns
    -------
    float
        P-value from log-rank test
    """
    try:
        result = multivariate_logrank_test(
            df[time_col], df[group_col], df[event_col]
        )
        return result.p_value
    except:
        return None


def compute_logrank_test(df, time_col, event_col, group_col, group1, group2):
    """
    Compute log-rank test p-value between two groups.
    
    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    event_col : str
    group_col : str
    group1, group2 : str
        Group identifiers to compare
    
    Returns
    -------
    float
        Log-rank test p-value
    """
    mask1 = df[group_col] == group1
    mask2 = df[group_col] == group2
    result = logrank_test(
        df[mask1][time_col], 
        df[mask2][time_col],
        df[mask1][event_col],
        df[mask2][event_col]
    )
    return result.p_value


def compute_pairwise_logrank(df, time_col, event_col, group_col):
    """
    Compute pairwise log-rank tests.
    
    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    event_col : str
    group_col : str
    
    Returns
    -------
    pd.DataFrame
        Pairwise p-values
    """
    try:
        return pairwise_logrank_test(df[time_col], df[group_col], df[event_col])
    except:
        return None


def compute_hr_and_ci(df, time_col, event_col, group_col, reference_group=None):
    """
    Compute HR and 95% CI using Cox regression.
    
    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    event_col : str
    group_col : str
    reference_group : str, optional
        Reference category
    
    Returns
    -------
    dict
        HR, CI_lower, CI_upper, p_value
    """
    try:
        df_cox = df[[group_col, time_col, event_col]].copy()
        df_cox = df_cox.dropna()
        
        if df_cox[group_col].dtype == 'object' or df_cox[group_col].dtype.name == 'category':
            categories = df_cox[group_col].unique()
            if len(categories) != 2:
                return None
            
            df_cox[group_col] = df_cox[group_col].astype('category')
            if reference_group:
                try:
                    df_cox[group_col] = df_cox[group_col].cat.set_categories(
                        [reference_group] + [c for c in categories if c != reference_group]
                    )
                except:
                    pass
        
        cph = CoxPHFitter()
        cph.fit(df_cox, duration_col=time_col, event_col=event_col)
        
        var_name = df_cox.columns[0]
        hr = np.exp(cph.params_[var_name])
        ci = np.exp(cph.confidence_intervals_.loc[var_name].values)
        p_value = cph.summary.loc[var_name, 'p']
        
        return {
            'HR': hr,
            'CI_lower': ci[0],
            'CI_upper': ci[1],
            'p_value': p_value
        }
    except:
        return None


def compute_cox_hr(df, time_col, event_col, group_col, reference_group=None):
    """
    Compute hazard ratio using univariable Cox regression.
    
    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    event_col : str
    group_col : str
    reference_group : str, optional
    
    Returns
    -------
    dict or None
        Dictionary with 'hr', 'ci_lower', 'ci_upper', 'p_value', 
        'reference', 'comparison' keys
    """
    try:
        cox_df = df[[time_col, event_col, group_col]].copy().dropna()
        groups = sorted(cox_df[group_col].unique())
        
        if len(groups) != 2:
            return None
        
        if reference_group and reference_group in groups:
            ref = reference_group
            comp = [g for g in groups if g != reference_group][0]
        else:
            ref, comp = groups[0], groups[1]
        
        cox_df['group_binary'] = (cox_df[group_col] == comp).astype(int)
        
        cph = CoxPHFitter()
        cph.fit(cox_df[[time_col, event_col, 'group_binary']], 
                duration_col=time_col, event_col=event_col)
        
        summary = cph.summary
        return {
            'hr': cph.hazard_ratios_['group_binary'],
            'ci_lower': summary['exp(coef) lower 95%']['group_binary'],
            'ci_upper': summary['exp(coef) upper 95%']['group_binary'],
            'p_value': summary['p']['group_binary'],
            'reference': ref,
            'comparison': comp
        }
    except:
        return None


def compute_median_survival(kmf):
    """
    Compute median survival as last observed time where survival >= 0.5.
    
    Parameters
    ----------
    kmf : KaplanMeierFitter
        Fitted Kaplan-Meier model
    
    Returns
    -------
    float
        Median survival time
    """
    try:
        sf = kmf.survival_function_
        times_above_half = sf[sf.iloc[:, 0] >= 0.5].index
        return times_above_half[-1] if len(times_above_half) > 0 else sf.index[0]
    except:
        return np.nan

