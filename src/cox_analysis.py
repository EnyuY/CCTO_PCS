"""Cox regression analysis module."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def load_and_prepare_data(filepath):
    """
    Load data and prepare for Cox regression analysis.
    
    Parameters
    ----------
    filepath : str
        Path to Excel data file
        
    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe ready for Cox regression
    """
    df = pd.read_excel(filepath, header=0)
    df.columns = df.columns.str.strip().str.replace('\xa0', ' ')
    
    print(f"Total samples: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    required_cols = ['Sex', 'Age', 'Pathological subtype', 'Baseline level of LDH',
                     'Conditions of Surgery', 'Adjuvant therapy(DFS)', 'Chemotherapy',
                     'First-line targeted therapy', 'Immunotherapy', 'Distant metastases',
                     'OS(months)', 'Dead status']
    
    df_analysis = df[required_cols].copy()
    df_analysis.columns = ['Sex', 'Age', 'Pathological_subtype', 'LDH',
                           'Surgery', 'Adjuvant', 'Chemotherapy',
                           'Targeted', 'Immunotherapy', 'Distant_metastases',
                           'OS_months', 'Event']
    
    return df_analysis


def encode_variables_for_cox(df):
    """
    Encode categorical variables for Cox regression.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe
        
    Returns
    -------
    tuple
        (df_encoded, variable_info)
    """
    df_cox = df.copy()
    variable_info = {}
    
    df_cox['Sex_male'] = (df_cox['Sex'].str.lower() == 'male').astype(int)
    variable_info['Sex'] = {'reference': 'Female', 'comparison': 'Male'}
    
    variable_info['Age'] = {'type': 'continuous', 'unit': 'years'}
    
    df_cox['LDH_elevated'] = (df_cox['LDH'].str.contains('Elevated', case=False, na=False)).astype(int)
    variable_info['LDH'] = {'reference': 'Normal', 'comparison': 'Elevated'}
    
    df_cox['Surgery_R1R2'] = (df_cox['Surgery'].str.contains('R1|R2', case=False, na=False)).astype(int)
    variable_info['Surgery'] = {'reference': 'R0', 'comparison': 'R1/R2'}
    
    variable_info['Distant_metastases'] = {'reference': 'No', 'comparison': 'Yes'}
    
    df_cox['Chemotherapy_yes'] = df_cox['Chemotherapy'].notna().astype(int)
    variable_info['Chemotherapy'] = {'reference': 'No', 'comparison': 'Yes'}
    
    df_cox['Targeted_1st'] = (df_cox['Targeted'].str.contains('1st', case=False, na=False)).astype(int)
    variable_info['Targeted'] = {'reference': '2nd line or above/No', 'comparison': '1st line'}
    
    df_cox['Immunotherapy_yes'] = df_cox['Immunotherapy'].notna().astype(int)
    variable_info['Immunotherapy'] = {'reference': 'No', 'comparison': 'Yes'}
    
    cox_vars = ['Sex_male', 'Age', 'LDH_elevated', 'Surgery_R1R2', 
                'Distant_metastases', 'Chemotherapy_yes', 'Targeted_1st', 
                'Immunotherapy_yes', 'OS_months', 'Event']
    
    df_cox_final = df_cox[cox_vars].dropna()
    
    print(f"\nSamples after removing missing values: {len(df_cox_final)}")
    print(f"Events (deaths): {df_cox_final['Event'].sum()}")
    
    return df_cox_final, variable_info


def univariate_cox_analysis(df_cox, variable_info):
    """
    Perform univariate Cox regression for each variable.
    
    Parameters
    ----------
    df_cox : pd.DataFrame
    variable_info : dict
        
    Returns
    -------
    pd.DataFrame
        Results of univariate analysis
    """
    results = []
    
    var_mapping = {
        'Sex_male': 'Sex',
        'Age': 'Age',
        'LDH_elevated': 'LDH',
        'Surgery_R1R2': 'Surgery',
        'Distant_metastases': 'Distant metastases',
        'Chemotherapy_yes': 'Chemotherapy',
        'Targeted_1st': 'First-line targeted therapy',
        'Immunotherapy_yes': 'Immunotherapy'
    }
    
    for var in var_mapping.keys():
        try:
            cph = CoxPHFitter()
            data = df_cox[[var, 'OS_months', 'Event']].copy()
            cph.fit(data, duration_col='OS_months', event_col='Event')
            
            hr = np.exp(cph.params_[var])
            ci_lower = np.exp(cph.confidence_intervals_.loc[var, '95% lower-bound'])
            ci_upper = np.exp(cph.confidence_intervals_.loc[var, '95% upper-bound'])
            p_value = cph.summary.loc[var, 'p']
            
            results.append({
                'Variable': var_mapping[var],
                'Encoded_var': var,
                'HR': hr,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper,
                'P_value': p_value
            })
            
        except Exception as e:
            print(f"Error in univariate analysis for {var}: {e}")
            continue
    
    return pd.DataFrame(results)


def multivariate_cox_analysis(df_cox, significant_vars=None, alpha=0.10):
    """
    Perform multivariate Cox regression.
    
    Parameters
    ----------
    df_cox : pd.DataFrame
    significant_vars : list, optional
    alpha : float
        
    Returns
    -------
    pd.DataFrame
        Results of multivariate analysis
    """
    var_mapping = {
        'Sex_male': 'Sex',
        'Age': 'Age',
        'LDH_elevated': 'LDH',
        'Surgery_R1R2': 'Surgery',
        'Distant_metastases': 'Distant metastases',
        'Chemotherapy_yes': 'Chemotherapy',
        'Targeted_1st': 'First-line targeted therapy',
        'Immunotherapy_yes': 'Immunotherapy'
    }
    
    if significant_vars is None:
        vars_to_use = list(var_mapping.keys())
    else:
        vars_to_use = significant_vars
    
    try:
        cph = CoxPHFitter()
        cols = vars_to_use + ['OS_months', 'Event']
        data = df_cox[cols].copy()
        cph.fit(data, duration_col='OS_months', event_col='Event')
        
        results = []
        for var in vars_to_use:
            hr = np.exp(cph.params_[var])
            ci_lower = np.exp(cph.confidence_intervals_.loc[var, '95% lower-bound'])
            ci_upper = np.exp(cph.confidence_intervals_.loc[var, '95% upper-bound'])
            p_value = cph.summary.loc[var, 'p']
            
            results.append({
                'Variable': var_mapping[var],
                'Encoded_var': var,
                'HR': hr,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper,
                'P_value': p_value
            })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"Error in multivariate analysis: {e}")
        return None

