"""Data loading and preprocessing."""

import pandas as pd


def load_full_data(filepath):
    """
    Load complete dataset from Excel file.
    
    Parameters
    ----------
    filepath : str
        Path to Excel data file (header in row 2)
    
    Returns
    -------
    pd.DataFrame
        Complete dataset
    """
    return pd.read_excel(filepath, header=1)


def load_and_preprocess_data(filepath, age_cutoff=28.5):
    """
    Load and preprocess data for Figure 3 analysis.
    
    Parameters
    ----------
    filepath : str
        Path to Excel data file
    age_cutoff : float
        Age threshold for stratification
        
    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe with age groups
    """
    df = pd.read_excel(filepath, header=1)
    
    cols = {
        'Age': df.columns[9],
        'Distant_metastases': df.columns[10],
        'Pathological_subtype': df.columns[11],
        'OS_months': df.columns[12],
        'Event': df.columns[14]
    }
    
    analysis_df = df[[cols['Age'], cols['Distant_metastases'], 
                      cols['Pathological_subtype'], cols['OS_months'], 
                      cols['Event']]].copy()
    analysis_df.columns = cols.keys()
    analysis_df = analysis_df.dropna(subset=['OS_months'])

    analysis_df['Age_group'] = analysis_df['Age'].apply(
        lambda x: 'Age >28' if x > age_cutoff else 'Age ≤28'
    )

    return analysis_df

