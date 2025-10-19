#!/usr/bin/env python3
"""
Prognostic factors analysis for cardiac sarcoma.
Performs univariate and multivariate Cox regression analysis.
Generates forest plot (Figure 4) and saves results to CSV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)


# ============================================================================
# Configuration
# ============================================================================

def set_nature_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['patch.linewidth'] = 0.5
    plt.rcParams['text.color'] = '#000000'
    plt.rcParams['axes.labelcolor'] = '#000000'
    plt.rcParams['xtick.color'] = '#000000'
    plt.rcParams['ytick.color'] = '#000000'
    plt.rcParams['axes.edgecolor'] = '#000000'


# ============================================================================
# Data loading and preprocessing
# ============================================================================

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
    # Read data (header in first row)
    df = pd.read_excel(filepath, header=0)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace('\xa0', ' ')
    
    # Print data overview
    print(f"Total samples: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Keep only necessary columns
    required_cols = ['Sex', 'Age', 'Pathological subtype', 'Baseline level of LDH',
                     'Conditions of Surgery', 'Adjuvant therapy(DFS)', 'Chemotherapy',
                     'First-line targeted therapy', 'Immunotherapy', 'Distant metastases',
                     'OS(months)', 'Dead status']
    
    df_analysis = df[required_cols].copy()
    
    # Rename for easier handling
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
        (df_encoded, variable_info) where variable_info contains encoding details
    """
    df_cox = df.copy()
    variable_info = {}
    
    # Sex: male=1, female=0
    df_cox['Sex_male'] = (df_cox['Sex'].str.lower() == 'male').astype(int)
    variable_info['Sex'] = {'reference': 'Female', 'comparison': 'Male'}
    
    # Age: continuous variable (already numeric)
    variable_info['Age'] = {'type': 'continuous', 'unit': 'years'}
    
    # LDH: Elevated=1, Normal=0
    df_cox['LDH_elevated'] = (df_cox['LDH'].str.contains('Elevated', case=False, na=False)).astype(int)
    variable_info['LDH'] = {'reference': 'Normal', 'comparison': 'Elevated'}
    
    # Surgery: R0=0, R1/R2=1
    df_cox['Surgery_R1R2'] = (df_cox['Surgery'].str.contains('R1|R2', case=False, na=False)).astype(int)
    variable_info['Surgery'] = {'reference': 'R0', 'comparison': 'R1/R2'}
    
    # Distant metastases: 0=no, 1=yes (already numeric)
    variable_info['Distant_metastases'] = {'reference': 'No', 'comparison': 'Yes'}
    
    # Chemotherapy: received=1, not received/NA=0
    df_cox['Chemotherapy_yes'] = df_cox['Chemotherapy'].notna().astype(int)
    variable_info['Chemotherapy'] = {'reference': 'No', 'comparison': 'Yes'}
    
    # Targeted therapy: 1st line=1, 2nd line or above/NA=0
    df_cox['Targeted_1st'] = (df_cox['Targeted'].str.contains('1st', case=False, na=False)).astype(int)
    variable_info['Targeted'] = {'reference': '2nd line or above/No', 'comparison': '1st line'}
    
    # Immunotherapy: received=1, not received/NA=0
    df_cox['Immunotherapy_yes'] = df_cox['Immunotherapy'].notna().astype(int)
    variable_info['Immunotherapy'] = {'reference': 'No', 'comparison': 'Yes'}
    
    # Select final columns for Cox regression
    cox_vars = ['Sex_male', 'Age', 'LDH_elevated', 'Surgery_R1R2', 
                'Distant_metastases', 'Chemotherapy_yes', 'Targeted_1st', 
                'Immunotherapy_yes', 'OS_months', 'Event']
    
    df_cox_final = df_cox[cox_vars].dropna()
    
    print(f"\nSamples after removing missing values: {len(df_cox_final)}")
    print(f"Events (deaths): {df_cox_final['Event'].sum()}")
    
    return df_cox_final, variable_info


# ============================================================================
# Cox regression analysis
# ============================================================================

def univariate_cox_analysis(df_cox, variable_info):
    """
    Perform univariate Cox regression for each variable.
    
    Parameters
    ----------
    df_cox : pd.DataFrame
        Encoded dataframe for Cox regression
    variable_info : dict
        Variable encoding information
        
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
        Encoded dataframe for Cox regression
    significant_vars : list, optional
        List of variables to include (if None, use all or select by p<alpha)
    alpha : float
        Significance level for variable selection
        
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
        # Use all variables
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


# ============================================================================
# Visualization: Forest plot
# ============================================================================

def format_pvalue(p):
    """Format p-value with appropriate precision."""
    if p < 0.001:
        return f"{p:.2e}"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def create_forest_plot(univariate_results, multivariate_results, output_dir='Figures'):
    """
    Create combined forest plot with shared axes.
    
    Parameters
    ----------
    univariate_results : pd.DataFrame
    multivariate_results : pd.DataFrame
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create single axis (compact height)
    n_vars = len(univariate_results)
    fig_height = max(2.5, n_vars * 0.2)
    fig, ax = plt.subplots(1, 1, figsize=(5, fig_height))
    
    # Plot combined forest plot
    plot_combined_forest(ax, univariate_results, multivariate_results)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure4.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/Figure4.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigure 4 saved to {output_dir}/")


def plot_combined_forest(ax, univariate_results, multivariate_results):
    """
    Plot combined forest plot with univariate and multivariate results.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    univariate_results : pd.DataFrame
    multivariate_results : pd.DataFrame
    """
    # Get all unique variables from univariate analysis
    variables = univariate_results['Variable'].tolist()
    n = len(variables)
    y_pos = np.arange(n)
    
    # Create mapping for multivariate results
    multi_dict = {}
    if multivariate_results is not None:
        for idx, row in multivariate_results.iterrows():
            multi_dict[row['Variable']] = {
                'HR': row['HR'],
                'CI_lower': row['CI_lower'],
                'CI_upper': row['CI_upper'],
                'P_value': row['P_value']
            }
    
    # Plot univariate results
    for i, (idx, row) in enumerate(univariate_results.iterrows()):
        hr = row['HR']
        ci_lower = row['CI_lower']
        ci_upper = row['CI_upper']
        p = row['P_value']
        
        # Color based on significance
        color = '#0173B2' if p < 0.05 else '#999999'
        
        # Plot CI line (offset slightly upward)
        y_offset = 0.12
        ax.plot([ci_lower, ci_upper], [i + y_offset, i + y_offset], 
                color=color, linewidth=1.2, zorder=1, alpha=0.8)
        
        # Plot HR point
        ax.scatter(hr, i + y_offset, s=40, color=color, zorder=2, 
                  edgecolors='white', linewidth=0.5, alpha=0.8)
    
    # Plot multivariate results
    for i, var in enumerate(variables):
        if var in multi_dict:
            multi = multi_dict[var]
            hr = multi['HR']
            ci_lower = multi['CI_lower']
            ci_upper = multi['CI_upper']
            p = multi['P_value']
            
            # Color based on significance
            color = '#0173B2' if p < 0.05 else '#999999'
            
            # Plot CI line (offset slightly downward)
            y_offset = -0.12
            ax.plot([ci_lower, ci_upper], [i + y_offset, i + y_offset], 
                    color=color, linewidth=1.5, zorder=3)
            
            # Plot HR point (diamond shape for multivariate)
            ax.scatter(hr, i + y_offset, s=50, color=color, zorder=4, 
                      marker='D', edgecolors='white', linewidth=0.5)
            
            # Add annotation for multivariate HR and p-value (closer to error bar)
            text = f"{hr:.2f} ({ci_lower:.2f}-{ci_upper:.2f}), p={format_pvalue(p)}"
            ax.text(hr, i + y_offset - 0.3, text, fontsize=5, 
                   ha='center', va='top', color='#000000')
    
    # Reference line at HR=1
    ax.axvline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Set y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables, fontsize=7, color='#000000')
    ax.set_ylim(-0.6, n - 0.4)
    
    # Set x-axis (log scale)
    ax.set_xscale('log')
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=8, color='#000000')
    ax.set_xlim(0.1, 15)
    
    # Add HR and p-value text for all univariate results (on the right side)
    for i, (idx, row) in enumerate(univariate_results.iterrows()):
        hr = row['HR']
        ci_lower = row['CI_lower']
        ci_upper = row['CI_upper']
        p = row['P_value']
        
        text = f"{hr:.2f} ({ci_lower:.2f}-{ci_upper:.2f}), p={format_pvalue(p)}"
        ax.text(1.02, i, text, transform=ax.get_yaxis_transform(),
               fontsize=6, va='center', ha='left', color='#000000')
    
    # Add legend (compact)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999', 
               markersize=5, label='Uni-COX', markeredgewidth=0.5, markeredgecolor='white'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#999999', 
               markersize=5, label='Multi-COX', markeredgewidth=0.5, markeredgecolor='white')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=6.5, 
             frameon=False, handletextpad=0.3, borderaxespad=0.3)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2, linewidth=0.5)


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Main analysis workflow."""
    set_nature_style()
    
    # Load data
    print("=" * 60)
    print("PROGNOSTIC FACTORS ANALYSIS")
    print("=" * 60)
    
    data_path = 'Data/20251019-R1-data-fac.xlsx'
    df = load_and_prepare_data(data_path)
    
    # Encode variables
    print("\n" + "=" * 60)
    print("ENCODING VARIABLES")
    print("=" * 60)
    df_cox, variable_info = encode_variables_for_cox(df)
    
    # Univariate analysis
    print("\n" + "=" * 60)
    print("UNIVARIATE COX REGRESSION")
    print("=" * 60)
    univariate_results = univariate_cox_analysis(df_cox, variable_info)
    print("\nUnivariate Results:")
    print(univariate_results.to_string(index=False))
    
    # Multivariate analysis (include variables with p<0.10 in univariate)
    print("\n" + "=" * 60)
    print("MULTIVARIATE COX REGRESSION")
    print("=" * 60)
    
    significant_vars = univariate_results[univariate_results['P_value'] < 0.10]['Encoded_var'].tolist()
    print(f"\nVariables with p<0.10 in univariate analysis: {len(significant_vars)}")
    
    if len(significant_vars) > 0:
        multivariate_results = multivariate_cox_analysis(df_cox, significant_vars)
        if multivariate_results is not None:
            print("\nMultivariate Results:")
            print(multivariate_results.to_string(index=False))
    else:
        print("\nNo variables reached p<0.10 in univariate analysis.")
        print("Performing multivariate analysis with all variables...")
        multivariate_results = multivariate_cox_analysis(df_cox)
        if multivariate_results is not None:
            print("\nMultivariate Results (all variables):")
            print(multivariate_results.to_string(index=False))
    
    # Save results to CSV
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    os.makedirs('Tables', exist_ok=True)
    univariate_results.to_csv('Tables/Figure4_univariate_analysis.csv', index=False)
    print("Univariate results saved to Tables/Figure4_univariate_analysis.csv")
    
    if multivariate_results is not None and len(multivariate_results) > 0:
        multivariate_results.to_csv('Tables/Figure4_multivariate_analysis.csv', index=False)
        print("Multivariate results saved to Tables/Figure4_multivariate_analysis.csv")
    
    # Create forest plot
    print("\n" + "=" * 60)
    print("GENERATING FOREST PLOT")
    print("=" * 60)
    create_forest_plot(univariate_results, multivariate_results)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

