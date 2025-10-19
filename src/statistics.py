"""Statistical analysis functions."""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
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


def compute_kaplan_meier_median(time_data, event_data):
    """
    Compute median survival using Kaplan-Meier method.
    
    Parameters
    ----------
    time_data : array-like
        Survival times
    event_data : array-like
        Event indicators (1=event, 0=censored)
    
    Returns
    -------
    float or None
        Median survival time
    """
    try:
        kmf = KaplanMeierFitter()
        kmf.fit(time_data, event_observed=event_data)
        
        survival_function = kmf.survival_function_
        median_candidates = survival_function[survival_function.iloc[:, 0] >= 0.5]
        
        if len(median_candidates) > 0:
            return median_candidates.index[-1]
        else:
            return None
    except:
        return None

