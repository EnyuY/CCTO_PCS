#!/usr/bin/env python3
"""
Survival analysis and figure generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
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
# Data loading
# ============================================================================

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
        Age threshold for stratification (default: 28.5)
        
    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe with age groups
    """
    df = pd.read_excel(filepath, header=1)
    
    # Extract relevant columns
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
    
    # Create age groups
    analysis_df['Age_group'] = analysis_df['Age'].apply(
        lambda x: 'Age >28' if x > age_cutoff else 'Age ≤28'
    )
    
    return analysis_df


# ============================================================================
# Statistical functions
# ============================================================================

def compute_logrank_test(df, time_col, event_col, group_col, group1, group2):
    """
    Compute log-rank test p-value between two groups.
    
    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
        Column name for time-to-event
    event_col : str
        Column name for event indicator (1=event, 0=censored)
    group_col : str
        Column name for group assignment
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


def compute_multivariate_logrank(df, time_col, event_col, group_col):
    """
    Compute multivariate log-rank test p-value for multiple groups.
    
    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    event_col : str
    group_col : str
        
    Returns
    -------
    float
        Multivariate log-rank test p-value
    """
    result = multivariate_logrank_test(
        df[time_col],
        df[group_col],
        df[event_col]
    )
    return result.p_value


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
        Reference group for HR calculation
        
    Returns
    -------
    dict or None
        Dictionary with 'hr', 'ci_lower', 'ci_upper', 'p_value', 
        'reference', 'comparison' keys, or None if not computable
    """
    try:
        cox_df = df[[time_col, event_col, group_col]].copy().dropna()
        groups = sorted(cox_df[group_col].unique())
        
        if len(groups) != 2:
            return None
        
        # Determine reference and comparison groups
        if reference_group and reference_group in groups:
            ref = reference_group
            comp = [g for g in groups if g != reference_group][0]
        else:
            ref, comp = groups[0], groups[1]
        
        # Encode: reference=0, comparison=1
        cox_df['group_binary'] = (cox_df[group_col] == comp).astype(int)
        
        # Fit Cox model
        cph = CoxPHFitter()
        cph.fit(cox_df[[time_col, event_col, 'group_binary']], 
                duration_col=time_col, event_col=event_col)
        
        # Extract results
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
        Median survival time (actual observed time point)
    """
    try:
        sf = kmf.survival_function_
        times_above_half = sf[sf.iloc[:, 0] >= 0.5].index
        return times_above_half[-1] if len(times_above_half) > 0 else sf.index[0]
    except:
        return np.nan


# ============================================================================
# Plotting functions
# ============================================================================

def format_pvalue(p):
    """Format p-value with appropriate precision."""
    if p < 0.001:
        # Scientific notation with 2 decimal places in coefficient
        return f"{p:.2e}"
    elif p < 0.01:
        # Regular format with 3 decimal places for 0.001 <= p < 0.01
        return f"{p:.3f}"
    else:
        # Regular format with 2 decimal places for p >= 0.01
        return f"{p:.2f}"


def add_number_at_risk(ax, kmf_dict, colors_dict, xlabel='Time (months)', time_points=None, y_offset=-0.10):
    """
    Add number at risk table below Kaplan-Meier plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    kmf_dict : dict
        Mapping of group labels to fitted KaplanMeierFitter objects
    colors_dict : dict
        Mapping of group labels to colors
    xlabel : str
        X-axis label to be placed below the table
    time_points : array-like, optional
        Time points for risk table (auto-generated if None)
    y_offset : float
        Vertical position offset below x-axis (increased spacing)
    """
    if time_points is None:
        all_times = []
        for kmf in kmf_dict.values():
            all_times.extend(kmf.event_table.index.tolist())
        max_time = max(all_times) if all_times else 100
        time_points = np.linspace(0, max_time, 5).astype(int)
    
    xlim = ax.get_xlim()
    x_range = xlim[1] - xlim[0]
    
    # Set x-axis ticks to match time_points exactly
    ax.set_xticks(time_points)
    ax.set_xticklabels([])
    
    # Add N@risk header as vertical text (not bold, same size as ylabel)
    # Positioned to align with "Survival (%)" and centered between risk numbers
    n_risk_y_center = y_offset - (len(kmf_dict) - 1) * 0.06 / 2  # Center of risk numbers
    ax.text(-0.12, n_risk_y_center - 0.01, 'N@risk',  # Slightly lower to center
            transform=ax.transAxes, fontsize=8, ha='center', 
            va='center', rotation=90, color='#000000')
    
    # Add rows for each group with color blocks and 0.75x spacing
    for i, (label, kmf) in enumerate(kmf_dict.items()):
        y_pos = y_offset - i * 0.06  # 0.75x spacing (0.08 * 0.75 = 0.06)
        
        # Get color for this group from the line
        line_color = None
        for line in ax.get_lines():
            if label in line.get_label():
                line_color = line.get_color()
                break
        if line_color is None:
            line_color = 'gray'
        
        # Color block instead of text label
        block_x = -0.08
        block_width = 0.04
        block_height = 0.02
        from matplotlib.patches import Rectangle
        rect = Rectangle((block_x, y_pos - block_height/2), block_width, block_height,
                         transform=ax.transAxes, facecolor=line_color, 
                         edgecolor='none', clip_on=False)
        ax.add_patch(rect)
        
        # Numbers at risk (pure black)
        for t in time_points:
            try:
                event_table_times = kmf.event_table.index[kmf.event_table.index <= t]
                n_risk = kmf.event_table.loc[event_table_times[-1], 'at_risk'] if len(event_table_times) > 0 else kmf.event_table.iloc[0]['at_risk']
            except:
                n_risk = 0
            
            x_pos = (t - xlim[0]) / x_range
            ax.text(x_pos, y_pos, str(int(n_risk)), transform=ax.transAxes,
                    fontsize=7, ha='center', va='center', color='#000000')
    
    # Add time point labels (pure black, increased spacing below, not bold)
    y_time = y_offset - len(kmf_dict) * 0.06 - 0.04  # Increased spacing
    for t in time_points:
        x_pos = (t - xlim[0]) / x_range
        ax.text(x_pos, y_time, str(int(t)), transform=ax.transAxes,
                fontsize=8, ha='center', va='center', color='#000000')
    
    # Add x-axis label below the table (increased spacing)
    y_xlabel = y_time - 0.06  # Increased spacing
    ax.text(0.5, y_xlabel, xlabel, transform=ax.transAxes,
            fontsize=8, ha='center', va='center', color='#000000')


def plot_km_curve_enhanced(ax, df, time_col, event_col, group_col=None, 
                          colors=None, label_map=None, show_hr=True, 
                          show_median=True, show_n_at_risk=True, min_n=2, 
                          reference_group=None, xlabel='Time (months)', title=''):
    """
    Generate Kaplan-Meier curve with statistical annotations.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df : pd.DataFrame
    time_col : str
        Column name for time-to-event
    event_col : str
        Column name for event indicator
    group_col : str, optional
        Column name for group stratification (None for single curve)
    colors : dict or list, optional
        Color mapping for groups
    label_map : dict, optional
        Custom labels for groups
    show_hr : bool
        Display hazard ratio and confidence interval
    show_median : bool
        Display median survival times
    show_n_at_risk : bool
        Display number at risk table
    min_n : int
        Minimum group size to include
    reference_group : str, optional
        Reference group for HR calculation
    xlabel : str
        X-axis label
    title : str
        Panel title
        
    Returns
    -------
    dict
        Statistics dictionary with results for each group
    """
    kmf = KaplanMeierFitter()
    stats_dict = {}
    kmf_dict = {}
    
    # Add horizontal line at y=0.5
    ax.axhline(0.5, color='grey', linestyle='--', alpha=0.4, linewidth=1)
    
    if group_col is None:
        # Single survival curve
        kmf.fit(df[time_col], event_observed=df[event_col], label='All')
        kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=1.5, 
                                   color=colors[0] if isinstance(colors, list) else '#1f77b4',
                                   show_censors=True, censor_styles={'marker': '+', 'ms': 4, 'mew': 1})
        median_surv = compute_median_survival(kmf)
        stats_dict['All'] = {'n': len(df), 'median': median_surv}
        kmf_dict['All'] = kmf
        
        if show_median and not np.isnan(median_surv):
            # Vertical line from y=0.5 to y=0
            ax.axvline(median_surv, ymin=0, ymax=0.5, color='grey', linestyle='--', alpha=0.6, linewidth=1)
            # Median value at bottom (no bold, pure black)
            ax.text(median_surv, 0.02, f'{median_surv:.2f}',
                    ha='center', fontsize=7, color='#000000')
    else:
        # Multiple groups
        df_clean = df.dropna(subset=[group_col])
        
        # Filter by minimum sample size
        if min_n > 1:
            group_counts = df_clean[group_col].value_counts()
            valid_groups = group_counts[group_counts >= min_n].index.tolist()
            df_clean = df_clean[df_clean[group_col].isin(valid_groups)]
        
        groups = sorted(df_clean[group_col].unique())
        
        if len(groups) == 0:
            return stats_dict
        
        # Assign colors
        if colors is None:
            if len(groups) == 2:
                colors = {groups[0]: '#0173B2', groups[1]: '#999999'}  #  blue and medium gray
            else:
                color_palette = plt.cm.Set2(np.linspace(0, 1, len(groups)))
                colors = {g: color_palette[i] for i, g in enumerate(groups)}
        
        # Plot each group and collect medians for smart positioning
        medians_list = []
        for group in groups:
            mask = df_clean[group_col] == group
            color = colors.get(group, colors.get(groups.index(group))) if isinstance(colors, dict) else colors[groups.index(group)]
            label = label_map.get(group, str(group)) if label_map else str(group)
            n = mask.sum()
            
            kmf_temp = KaplanMeierFitter()
            kmf_temp.fit(df_clean[mask][time_col], 
                        event_observed=df_clean[mask][event_col], 
                        label=label)
            kmf_temp.plot_survival_function(ax=ax, ci_show=False, 
                                           linewidth=1.5, color=color,
                                           show_censors=True, censor_styles={'marker': '+', 'ms': 4, 'mew': 1})
            
            median_surv = compute_median_survival(kmf_temp)
            stats_dict[group] = {'n': n, 'median': median_surv, 'label': label, 'color': color}
            kmf_dict[label] = kmf_temp
            medians_list.append((median_surv, color))
        
        # Sort medians to position labels intelligently (no bold)
        if show_median:
            sorted_medians = sorted([(m, c) for m, c in medians_list if not np.isnan(m)])
            for i, (median_surv, color) in enumerate(sorted_medians):
                # Vertical line from y=0.5 to y=0
                ax.axvline(median_surv, ymin=0, ymax=0.5, color=color, linestyle='--', alpha=0.6, linewidth=1)
                # Position text: smaller values on left, larger on right (no bold)
                if i < len(sorted_medians) / 2:
                    # Left side of line
                    ax.text(median_surv - 0.5, 0.02, f'{median_surv:.2f}',
                            ha='right', fontsize=7, color='#000000')
                else:
                    # Right side of line
                    ax.text(median_surv + 0.5, 0.02, f'{median_surv:.2f}',
                            ha='left', fontsize=7, color='#000000')
        
        # Statistics for two-group comparison
        if len(groups) == 2:
            pval = compute_logrank_test(df_clean, time_col, event_col, 
                                       group_col, groups[0], groups[1])
            hr_results = compute_cox_hr(df_clean, time_col, event_col, 
                                       group_col, reference_group=reference_group)
            
            stats_dict['p_value'] = pval
            stats_dict['hr_results'] = hr_results
            
            # Legend at upper right (no frame), moved to top right corner
            legend = ax.legend(loc='upper right', frameon=False, fontsize=8, 
                             prop={'family': 'sans-serif', 'size': 8},
                             bbox_to_anchor=(1.1, 1.1))
            
            # Title and HR - title normal, HR condensed, moved up and right
            legend_y = 0.82  # Moved up
            if title:
                # Title with normal width
                ax.text(1.07, legend_y, title,
                        transform=ax.transAxes, fontsize=8, ha='right', va='top', 
                        color='#000000', family='sans-serif')
                legend_y -= 0.07  # Increased spacing between title and HR
            
            if show_hr and hr_results:
                # HR with condensed/narrow font
                hr_text = f"HR {hr_results['hr']:.2f}({hr_results['ci_lower']:.2f}-{hr_results['ci_upper']:.2f})"
                ax.text(1.07, legend_y, hr_text,
                        transform=ax.transAxes, fontsize=8, ha='right', va='top', 
                        color='#000000', family='sans-serif', stretch='condensed')
            
            # Add p-value at y=0.2
            p_text = f'p = {format_pvalue(pval)}'
            ax.text(0.05, 0.20, p_text,
                    transform=ax.transAxes, fontsize=8, ha='left', va='bottom', color='#000000')
        
        elif len(groups) > 2:
            # Multivariate comparison
            try:
                pval = compute_multivariate_logrank(df_clean, time_col, 
                                                   event_col, group_col)
                stats_dict['p_value'] = pval
                
                # Legend at upper right (no frame), moved to top right corner
                legend = ax.legend(loc='upper right', frameon=False, fontsize=8,
                                 prop={'family': 'sans-serif', 'size': 8},
                                 bbox_to_anchor=(1.1, 1.1))
                
                # Title moved up and right
                if title:
                    ax.text(1.07, 0.75, title,
                            transform=ax.transAxes, fontsize=8, ha='right', va='top', 
                            color='#000000', family='sans-serif')
                
                # Add p-value at y=0.2
                p_text = f'p = {format_pvalue(pval)}'
                ax.text(0.05, 0.20, p_text,
                        transform=ax.transAxes, fontsize=8, ha='left', va='bottom', color='#000000')
            except:
                pass
    
    # Add number at risk table
    if show_n_at_risk and len(kmf_dict) > 0:
        add_number_at_risk(ax, kmf_dict, colors, xlabel=xlabel)
    
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('Survival (%)', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return stats_dict


def add_panel_labels(axes_flat, labels=None):
    """
    Add panel labels (A, B, C, ...) to subplot axes.
    
    Parameters
    ----------
    axes_flat : array-like
        Flattened array of axes objects
    labels : list, optional
        Custom labels (default: A, B, C, ...)
    """
    if labels is None:
        labels = [chr(65 + i) for i in range(len(axes_flat))]
    
    for ax_obj, label in zip(axes_flat, labels):
        ax_obj.text(-0.15, 1.05, label, transform=ax_obj.transAxes, 
                    fontsize=11, va='top', ha='right', color='black')


# ============================================================================
# Figure 3 specific plotting functions (legacy)
# ============================================================================

def plot_scatter_age_os(ax, df, age_cutoff, colors):
    """Plot scatter: Age vs OS with correlation (flattened more)."""
    for age_group in ['Age ≤28', 'Age >28']:
        subset = df[df['Age_group'] == age_group]
        ax.scatter(subset['Age'], subset['OS_months'], 
                   c=colors[age_group], label=age_group,
                   alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
    
    ax.axvline(age_cutoff, color='gray', linestyle='--', 
               linewidth=0.5, alpha=0.5)
    
    from scipy.stats import spearmanr
    corr, corr_pval = spearmanr(df['Age'], df['OS_months'])
    # P value with line break and lowercase p
    ax.text(0.05, 0.95, f'Spearman r = {corr:.2f}\np = {corr_pval:.3f}', 
            transform=ax.transAxes, fontsize=7, va='top', color='#000000')
    
    # Remove default xlabel
    ax.set_xlabel('')
    ax.set_ylabel('OS (months)', fontsize=8, color='#000000')
    
    # Add x-axis label centered below plot
    ax.text(0.5, -0.25, 'Age (years)', transform=ax.transAxes,
            fontsize=8, ha='center', va='top', color='#000000')
    
    # Add legend centered below x-axis label, using scatter markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Age ≤28'], 
               markersize=6, markeredgecolor='white', markeredgewidth=0.5, label='≤28'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Age >28'], 
               markersize=6, markeredgecolor='white', markeredgewidth=0.5, label='>28')
    ]
    leg = ax.legend(handles=legend_elements, fontsize=7, frameon=False, 
                    loc='lower center', bbox_to_anchor=(0.5, -0.30),  # Moved up slightly
                    ncol=2, columnspacing=1.5, handletextpad=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)




def plot_km_pathological_subtype(ax, df, time_col, event_col, subtype_col, 
                                  xlabel='OS(Months)', title='', min_n=2):
    """    
    Parameters
    ----------
    ax : matplotlib axes
    df : pd.DataFrame
    time_col : str
    event_col : str
    subtype_col : str
    xlabel : str
    title : str
    min_n : int
        Minimum sample size for subtype inclusion
        
    Returns
    -------
    tuple
        (major_subtypes list, p_value)
    """
    subtype_counts = df[subtype_col].value_counts()
    major_subtypes = subtype_counts[subtype_counts >= min_n].index.tolist()
    
    # Assign colors
    color_palette = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', 
                     '#949494', '#FBB4AE', '#B3CDE3']
    colors_dict = {subtype: color_palette[i % len(color_palette)] 
                   for i, subtype in enumerate(major_subtypes)}
    
    # Plot KM curves
    kmf = KaplanMeierFitter()
    kmf_dict = {}
    
    for subtype in major_subtypes:
        mask = df[subtype_col] == subtype
        n = mask.sum()
        kmf_sub = KaplanMeierFitter()
        kmf_sub.fit(df[mask][time_col], 
                    event_observed=df[mask][event_col], 
                    label=subtype)
        kmf_sub.plot_survival_function(
            ax=ax, ci_show=False, linewidth=1.5, 
            color=colors_dict[subtype],
            show_censors=True, 
            censor_styles={'marker': '+', 'ms': 4, 'mew': 1}
        )
        kmf_dict[subtype] = kmf_sub
    
    # Calculate multivariate log-rank p-value
    try:
        df_clean = df[df[subtype_col].isin(major_subtypes)]
        p_val = compute_multivariate_logrank(df_clean, time_col, event_col, subtype_col)
    except:
        p_val = None
    
    # Axis styling
    ax.set_xlim(left=0)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('Survival probability', fontsize=8, color='#000000')
    ax.set_xlabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend - no (n=xxx)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, major_subtypes, 
              loc='lower left', frameon=False, 
              prop={'family': 'sans-serif', 'size': 7},
              bbox_to_anchor=(0.02, 0.02))
    
    # Title at top-right
    if title:
        ax.text(1.07, 0.75, title,
                transform=ax.transAxes, fontsize=8, ha='right', va='top',
                color='#000000', family='sans-serif')
    
    # P-value annotation
    if p_val is not None:
        p_text = f'p={format_pvalue(p_val)}'
        ax.text(0.05, 0.20, p_text,
                transform=ax.transAxes, fontsize=8, ha='left', va='top',
                color='#000000', family='sans-serif')
    
    # Number at Risk
    add_number_at_risk(ax, kmf_dict, colors_dict, xlabel)
    
    return major_subtypes, p_val


def plot_boxplot_with_points(ax, df, value_col, group_col, group_order=None, colors_dict=None):
    """    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df : pd.DataFrame
    value_col : str
    group_col : str
    group_order : list, optional
    colors_dict : dict, optional
        Mapping of group labels to colors (for points)
    
    Returns
    -------
    tuple
        (data_list, labels_list, positions_list)
    """
    if group_order is None:
        group_order = sorted(df[group_col].unique())
    
    data_list = []
    positions_list = []
    labels_list = []
    
    for i, group in enumerate(group_order):
        subset = df[df[group_col] == group][value_col]
        if len(subset) > 0:
            data_list.append(subset)
            positions_list.append(i)
            labels_list.append(group)
    
    # Create boxplot - no fill color,  style
    ax.boxplot(data_list, positions=positions_list, widths=0.5, 
               patch_artist=True,
               boxprops=dict(facecolor='none', edgecolor='#000000', linewidth=0.8),
               medianprops=dict(color='#000000', linewidth=1.2),
               whiskerprops=dict(color='#000000', linewidth=0.8),
               capprops=dict(color='#000000', linewidth=0.8),
               flierprops=dict(marker='o', markerfacecolor='none', 
                             markeredgecolor='#000000', markersize=4, linewidth=0.8))
    
    # Overlay points with colors matching survival curves
    for pos, data, label in zip(positions_list, data_list, labels_list):
        y = data.values
        x = np.random.normal(pos, 0.08, size=len(y))
        
        # Determine point color
        if colors_dict and label in colors_dict:
            point_color = colors_dict[label]
        else:
            # Default to  blue for single/multiple groups
            point_color = '#0173B2'
        
        ax.scatter(x, y, alpha=0.7, s=25, color=point_color, 
                  edgecolors='white', linewidth=0.5, zorder=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return data_list, labels_list, positions_list


def add_statistical_annotation(ax, pval, text_template, x=0.05, y=0.05, bbox=True):
    """Add statistical annotation text to plot."""
    text = text_template.format(pval=pval)
    if bbox:
        ax.text(x, y, text, transform=ax.transAxes, fontsize=7, 
                bbox=dict(boxstyle='round', facecolor='white', 
                         alpha=0.8, edgecolor='none'))
    else:
        ax.text(x, y, text, transform=ax.transAxes, fontsize=7)


# ============================================================================
# Figure generation functions
# ============================================================================

def generate_figure1(df_full, output_dir='Figures', table_dir='Tables'):
    """
    Generate Figure 1: Patient characteristics and prognostic factors.
    
    Panels:
    A: Pathological subtype vs OS
    B: Baseline LDH vs OS
    C: Cox regression forest plot (prognostic factors)
    
    Parameters
    ----------
    df_full : pd.DataFrame
    output_dir : str
    table_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    
    # Import Cox analysis functions from Figure1C_COX script
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from Figure1C_COX import (
        load_and_prepare_data, 
        encode_variables_for_cox,
        univariate_cox_analysis,
        multivariate_cox_analysis,
        plot_combined_forest
    )
    
    # Prepare data
    os_col = df_full.columns[12]
    df_analysis = df_full[[df_full.columns[1], df_full.columns[2], 
                           os_col, df_full.columns[14]]].copy()
    df_analysis.columns = ['Pathological_subtype', 'LDH', 'OS_months', 'Event']
    df_analysis = df_analysis.dropna(subset=['OS_months'])
    
    # Create figure with GridSpec for custom layout
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(7.08, 4.8))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1, 1], 
                  hspace=0.50, wspace=0.0)
    
    # Top row: A and B (survival curves, maintain original height)
    ax_a = fig.add_subplot(gs[0, :2])  # Left 2/5
    ax_b = fig.add_subplot(gs[0, 3:])  # Right 2/5 (with gap)
    
    # Bottom row: C (forest plot; occupy center 3/5, leave 1/5 blank on both sides)
    ax_c = fig.add_subplot(gs[1, 1:4])  # Center 3/5, flanked by empty 1/5 on both sides
    
    # Panel A: Pathological subtype
    stats_a = plot_km_curve_enhanced(
        ax_a, df_analysis, 'OS_months', 'Event', 
        group_col='Pathological_subtype',
        show_hr=True, show_median=True, show_n_at_risk=True, min_n=2,
        xlabel=os_col,
        title='Pathological subtype'
    )
    
    # Panel B: LDH
    stats_b = plot_km_curve_enhanced(
        ax_b, df_analysis, 'OS_months', 'Event',
        group_col='LDH',
        show_hr=True, show_median=True, show_n_at_risk=True, min_n=2,
        reference_group='elevated',
        xlabel=os_col,
        title='Baseline level of LDH'
    )
    
    # Panel C: Cox regression forest plot
    print("\nGenerating Panel C: Cox regression analysis...")
    data_path = 'Data/20251019-R1-data-fac.xlsx'
    df_cox_raw = load_and_prepare_data(data_path)
    df_cox, variable_info = encode_variables_for_cox(df_cox_raw)
    
    # Univariate analysis
    univariate_results = univariate_cox_analysis(df_cox, variable_info)
    
    # Multivariate analysis
    significant_vars = univariate_results[univariate_results['P_value'] < 0.10]['Encoded_var'].tolist()
    if len(significant_vars) > 0:
        multivariate_results = multivariate_cox_analysis(df_cox, significant_vars)
    else:
        multivariate_results = None
    
    # Plot forest plot in panel C
    plot_combined_forest(ax_c, univariate_results, multivariate_results)
    
    # Add panel labels
    ax_a.text(-0.20, 1.05, 'A', transform=ax_a.transAxes, 
             fontsize=11, va='top', ha='right', color='black')
    ax_b.text(-0.15, 1.05, 'B', transform=ax_b.transAxes, 
             fontsize=11, va='top', ha='right', color='black')
    ax_c.text(-0.3, 1.05, 'C', transform=ax_c.transAxes, 
             fontsize=11, va='top', ha='right', color='black')
    
    # Save
    plt.savefig(f'{output_dir}/Figure1.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/Figure1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics
    stats_data = []
    for var, stats_dict in [('Pathological_subtype', stats_a), ('LDH', stats_b)]:
        for key, val in stats_dict.items():
            if key not in ['p_value', 'hr_results'] and isinstance(val, dict):
                row = {
                    'Variable': var,
                    'Group': val.get('label', key),
                    'N': val.get('n', ''),
                    'Median_OS': val.get('median', ''),
                    'P_value': stats_dict.get('p_value', '')
                }
                hr_res = stats_dict.get('hr_results')
                if hr_res:
                    row.update({'HR': hr_res['hr'], 
                               'CI_lower': hr_res['ci_lower'],
                               'CI_upper': hr_res['ci_upper']})
                stats_data.append(row)
    
    pd.DataFrame(stats_data).to_csv(f'{table_dir}/Figure1_statistics.csv', 
                                    index=False)
    
    # Save Cox regression results
    univariate_results.to_csv(f'{table_dir}/Figure1C_univariate_analysis.csv', index=False)
    if multivariate_results is not None and len(multivariate_results) > 0:
        multivariate_results.to_csv(f'{table_dir}/Figure1C_multivariate_analysis.csv', index=False)
    
    print("Figure1.pdf/png saved")
    print("Figure1_statistics.csv saved")
    print("Figure1C Cox analysis CSVs saved")
    
    return {'stats_a': stats_a, 'stats_b': stats_b, 
            'univariate': univariate_results, 'multivariate': multivariate_results}


def generate_figure2(df_full, output_dir='Figures', table_dir='Tables'):
    """
    Generate Figure 2: Treatment outcomes survival analysis.
    
    Panels:
    A: All patients
    B: Surgical conditions  
    C: Adjuvant therapy (DFS)
    D: Chemotherapy
    E: Targeted therapy
    F: Immunotherapy
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    
    dead_status_col = df_full.columns[14]
    
    # Column names for x-axis labels
    col_names = {
        'os': df_full.columns[12],
        'dfs': df_full.columns[13],
        'surgery': df_full.columns[16],
        'chemo': df_full.columns[17],
        'targeted': df_full.columns[18],
        'immuno': df_full.columns[19]
    }
    
    # Create figure - 3 rows x 2 columns, wider panels, flatter, increased spacing
    fig, axes = plt.subplots(3, 2, figsize=(7.08, 7.0))
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.08, top=0.98, hspace=0.5, wspace=0.40)  # Triple spacing
    all_stats = {}
    
    # Panel A: All patients
    df_all = df_full[[df_full.columns[12], dead_status_col]].copy()
    df_all.columns = ['OS_months', 'Event']
    df_all = df_all.dropna()
    stats_a = plot_km_curve_enhanced(
        axes[0, 0], df_all, 'OS_months', 'Event',
        group_col=None, colors=['#1f77b4'],
        show_hr=False, show_median=True, show_n_at_risk=True,
        xlabel=col_names['os'],
        title='All patients'
    )
    all_stats['All'] = stats_a
    
    # Panel B: Surgery
    df_surgery = df_full[[df_full.columns[4], df_full.columns[16], 
                          dead_status_col]].copy()
    df_surgery.columns = ['Surgery', 'OS_months', 'Event']
    df_surgery = df_surgery.dropna()
    stats_b = plot_km_curve_enhanced(
        axes[0, 1], df_surgery, 'OS_months', 'Event',
        group_col='Surgery', show_hr=True, show_median=True, 
        show_n_at_risk=True, reference_group='R1/R2',
        xlabel=col_names['surgery'],
        title='Surgical conditions'
    )
    all_stats['Surgery'] = stats_b
    
    # Panel C: Adjuvant therapy (DFS)
    df_adjuvant = df_full[[df_full.columns[5], df_full.columns[13], 
                           dead_status_col]].copy()
    df_adjuvant.columns = ['Adjuvant', 'DFS_months', 'Event']
    df_adjuvant = df_adjuvant.dropna()
    stats_c = plot_km_curve_enhanced(
          axes[1, 0], df_adjuvant, 'DFS_months', 'Event',
          group_col='Adjuvant', show_hr=True, show_median=True, 
          show_n_at_risk=True, reference_group='no adjuvant chemotherapy',
          xlabel=col_names['dfs'],
          title='Adjuvant therapy'
    )
    all_stats['Adjuvant'] = stats_c
    
    # Panel D: Chemotherapy
    df_chemo = df_full[[df_full.columns[6], df_full.columns[17], 
                        dead_status_col]].copy()
    df_chemo.columns = ['Chemo', 'OS_months', 'Event']
    df_chemo = df_chemo.dropna()
    stats_d = plot_km_curve_enhanced(
        axes[1, 1], df_chemo, 'OS_months', 'Event',
        group_col='Chemo', show_hr=True, show_median=True, 
        show_n_at_risk=True,
        xlabel=col_names['chemo'],
        title='Chemotherapy'
    )
    all_stats['Chemotherapy'] = stats_d
    
    # Panel E: Targeted therapy
    df_targeted = df_full[[df_full.columns[7], df_full.columns[18], 
                           dead_status_col]].copy()
    df_targeted.columns = ['Targeted', 'OS_months', 'Event']
    df_targeted = df_targeted.dropna(subset=['OS_months', 'Targeted'])
    stats_e = plot_km_curve_enhanced(
        axes[2, 0], df_targeted, 'OS_months', 'Event',
        group_col='Targeted', show_hr=True, show_median=True, 
        show_n_at_risk=True, reference_group='2nd line or above',
        xlabel=col_names['targeted'],
        title='Targeted therapy'
    )
    all_stats['Targeted'] = stats_e
    
    # Panel F: Immunotherapy
    df_immuno = df_full[[df_full.columns[8], df_full.columns[19], 
                         dead_status_col]].copy()
    df_immuno.columns = ['Immuno', 'OS_months', 'Event']
    df_immuno = df_immuno.dropna(subset=['OS_months', 'Immuno'])
    stats_f = plot_km_curve_enhanced(
        axes[2, 1], df_immuno, 'OS_months', 'Event',
        group_col='Immuno', show_hr=True, show_median=True, 
        show_n_at_risk=True, reference_group='2nd line or above',
        xlabel=col_names['immuno'],
        title='Immunotherapy'
    )
    all_stats['Immunotherapy'] = stats_f
    
    add_panel_labels(axes.flat, ['A', 'B', 'C', 'D', 'E', 'F'])
    
    # Save
    plt.savefig(f'{output_dir}/Figure2.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/Figure2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics
    stats_data = []
    for treatment, stats_dict in all_stats.items():
        for key, val in stats_dict.items():
            if key not in ['p_value', 'hr_results'] and isinstance(val, dict):
                row = {
                    'Treatment': treatment,
                    'Group': val.get('label', key),
                    'N': val.get('n', ''),
                    'Median_OS': val.get('median', ''),
                    'P_value': stats_dict.get('p_value', ''),
                }
                hr_res = stats_dict.get('hr_results')
                if hr_res:
                    row.update({'HR': hr_res['hr'],
                               'CI_lower': hr_res['ci_lower'],
                               'CI_upper': hr_res['ci_upper']})
                stats_data.append(row)
    
    pd.DataFrame(stats_data).to_csv(f'{table_dir}/Figure2_statistics.csv', 
                                    index=False)
    
    return all_stats


def generate_figure3(df, output_dir='Figures', table_dir='Tables'):
    """
    Generate Figure 3: Age, metastases, and pathological subtype analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    
    age_cutoff = 28.5
    colors_age = {'Age ≤28': '#0173B2', 'Age >28': '#999999'}
    colors_met = {'No metastases': '#0173B2', 'Metastases': '#999999'}
    
    # Create figure with adjusted layout
    # Row 1: A (narrow) + B (standard width like Fig1/2)
    # Row 2: C (narrow, aligned with A) + D (standard width like Fig1/2)
    # Row 3: E (same total width as A+B or C+D)
    fig = plt.figure(figsize=(7.08, 7.0))
    gs = fig.add_gridspec(3, 10, left=0.08, right=0.98, bottom=0.08, top=0.98, 
                          hspace=0.50, wspace=0.48,  # Increased wspace by 1.2x (0.40 * 1.2 = 0.48)
                          height_ratios=[1, 1, 1.2])
    
    # Create axes with adjusted width ratios
    # A: 2 units (narrow), B: 5 units (standard)
    # C: 2 units (narrow, aligned with A), D: 5 units (standard)
    # E: same width as A+B (2 + spacing + 5 = 8 units including spacing)
    ax_00 = fig.add_subplot(gs[0, :2])    # A: Age scatter (narrow)
    ax_01 = fig.add_subplot(gs[0, 3:8])   # B: Age KM (standard width)
    ax_10 = fig.add_subplot(gs[1, :2])    # C: Metastases boxplot (narrow, aligned with A)
    ax_11 = fig.add_subplot(gs[1, 3:8])   # D: Metastases KM (standard width)
    ax_20 = fig.add_subplot(gs[2, :8])    # E: Pathological subtype boxplot (same width as top rows)
    
    # Row 1, Col 1: Age vs OS scatter (A - narrow)
    plot_scatter_age_os(ax_00, df, age_cutoff, colors_age)
    
    # Row 1, Col 2: KM by age (B - standard width)
    plot_km_curve_enhanced(
        ax_01, df, 'OS_months', 'Event',
        group_col='Age_group', show_hr=True, show_median=True,
        show_n_at_risk=True, xlabel='OS(Months)',
        title='Age groups'
    )
    
    # Row 2, Col 1: Boxplot by metastases (C - narrow, aligned with A)
    # Create color mapping for boxplot points
    met_colors_for_box = {0: '#0173B2', 1: '#999999'}
    data_list, labels_list, positions_list = plot_boxplot_with_points(
        ax_10, df, 'OS_months', 'Distant_metastases', 
        group_order=[0, 1], colors_dict=met_colors_for_box
    )
    n0 = (df['Distant_metastases'] == 0).sum()
    n1 = (df['Distant_metastases'] == 1).sum()
    ax_10.set_xticks([0, 1])
    ax_10.set_xticklabels([f'No\n(n={n0})', f'Yes\n(n={n1})'], fontsize=7, color='#000000')
    ax_10.set_ylabel('OS (months)', fontsize=8, color='#000000')
    ax_10.set_xlabel('Distant metastases', fontsize=8, color='#000000')
    
    mannwhitney_pval = stats.mannwhitneyu(data_list[0], data_list[1])[1]
    
    # Add significance bracket (small hat) first
    if mannwhitney_pval < 0.05:
        y_max = max([d.max() for d in data_list])
        y_range = ax_10.get_ylim()[1] - ax_10.get_ylim()[0]
        y_bracket = y_max + y_range * 0.05
        ax_10.plot([0, 0, 1, 1], 
                   [y_bracket, y_bracket + y_range * 0.02, 
                    y_bracket + y_range * 0.02, y_bracket], 
                   'k-', linewidth=0.8)
        
        # Add p-value and significance marker ABOVE the bracket
        p_text = f'p = {format_pvalue(mannwhitney_pval)}'
        if mannwhitney_pval < 0.01:
            sig_marker = '**'
        else:
            sig_marker = '*'
        p_text += f' {sig_marker}'
        
        ax_10.text(0.5, y_bracket + y_range * 0.04, p_text, 
                   fontsize=8, ha='center', va='bottom', color='#000000')
    else:
        # If not significant, show p-value at top
        p_text = f'p = {format_pvalue(mannwhitney_pval)}'
        ax_10.text(0.5, 0.95, p_text, transform=ax_10.transAxes, 
                   fontsize=8, ha='center', va='top', color='#000000')
    
    # Row 2, Col 2: KM by metastases (D - standard width)
    df_met = df.copy()
    df_met['Metastases_label'] = df_met['Distant_metastases'].map({
        0: 'No metastases', 
        1: 'Metastases'
    })
    plot_km_curve_enhanced(
        ax_11, df_met, 'OS_months', 'Event',
        group_col='Metastases_label', show_hr=True, show_median=True,
        show_n_at_risk=True, xlabel='OS(Months)',
        title='Distant metastases'
    )
    
    # Row 3: Boxplot by pathological subtype (E - same width as top rows)
    subtype_order = df.groupby('Pathological_subtype')['OS_months'].median().sort_values(
        ascending=False).index.tolist()
    
    data_list = []
    labels_list = []
    positions_list = []
    
    # Abbreviations for subtypes
    abbrev = {
        'Epithelioid Hemangioendothelioma': 'EHE',
        'Myxoid Fibrosarcoma': 'MFS',
        'Intimal Sarcoma': 'IS',
        'Leiomyosarcoma': 'LMS',
        'Angiosarcoma': 'AS',
        'Synovial Sarcoma': 'SS',
        'Rhabdomyosarcoma': 'RMS',
        'Osteosarcoma': 'OS'
    }
    
    # Use abbreviations for x-axis labels
    for i, subtype in enumerate(subtype_order):
        subset = df[df['Pathological_subtype'] == subtype]['OS_months']
        if len(subset) > 0:
            data_list.append(subset)
            positions_list.append(i)
            short_label = abbrev.get(subtype, subtype)
            labels_list.append(f'{short_label}\n(n={len(subset)})')
    
    ax_20.boxplot(data_list, positions=positions_list, widths=0.5, 
                  patch_artist=True,
                  boxprops=dict(facecolor='none', edgecolor='#000000', linewidth=0.8),
                  medianprops=dict(color='#000000', linewidth=1.2),
                  whiskerprops=dict(color='#000000', linewidth=0.8),
                  capprops=dict(color='#000000', linewidth=0.8),
                  flierprops=dict(marker='o', markerfacecolor='none', 
                                markeredgecolor='#000000', markersize=4, linewidth=0.8))
    
    for i, (pos, data) in enumerate(zip(positions_list, data_list)):
        y = data.values
        x = np.random.normal(pos, 0.08, size=len(y))
        ax_20.scatter(x, y, alpha=0.7, s=25, color='#0173B2', 
                     edgecolors='white', linewidth=0.5, zorder=10)
    
    ax_20.set_xticks(positions_list)
    # Use abbreviations, no rotation
    ax_20.set_xticklabels(labels_list, fontsize=7, rotation=0, ha='center', color='#000000')
    ax_20.set_ylabel('OS (months)', fontsize=8, color='#000000')
    ax_20.set_xlabel('Pathological subtype', fontsize=8, color='#000000')
    ax_20.spines['top'].set_visible(False)
    ax_20.spines['right'].set_visible(False)
    
    # Statistical tests
    kw_pval = None
    if len(data_list) > 2:
        h_stat, kw_pval = stats.kruskal(*data_list)
        kw_text = f'Kruskal-Wallis p = {format_pvalue(kw_pval)}'
        # KW p-value at upper right
        ax_20.text(0.98, 0.98, kw_text, transform=ax_20.transAxes, 
                   fontsize=8, ha='right', va='top', color='#000000')
    
    # Add abbreviation legend below KW p-value (right-aligned, with extra line break)
    abbrev_text_lines = []
    for subtype in subtype_order:
        subset = df[df['Pathological_subtype'] == subtype]['OS_months']
        if len(subset) > 0:
            short = abbrev.get(subtype, subtype)
            abbrev_text_lines.append(f'{subtype} ({short})')
    
    abbrev_text = '\n' + '\n'.join(abbrev_text_lines)  # Add extra \n at beginning
    ax_20.text(0.98, 0.90, abbrev_text, transform=ax_20.transAxes, 
               fontsize=6, ha='right', va='top', color='#000000',
               linespacing=1.3)
    
    # Pairwise comparisons for all pairs with significance markers
    from itertools import combinations
    y_max_base = max([d.max() for d in data_list])
    y_range = y_max_base - min([d.min() for d in data_list])
    
    pairwise_results = []
    for i, j in combinations(range(len(data_list)), 2):
        if len(data_list[i]) >= 3 and len(data_list[j]) >= 3:
            u_stat, p_val = stats.mannwhitneyu(data_list[i], data_list[j])
            pairwise_results.append((positions_list[i], positions_list[j], p_val, i, j))
    
    # Draw ALL pairwise comparisons with small hats (not just significant ones)
    y_offset_step = y_range * 0.08
    for bracket_idx, (pos1, pos2, p_val, i, j) in enumerate(pairwise_results):
        y_line = y_max_base + y_range * 0.05 + bracket_idx * y_offset_step
        
        # Small hat bracket
        ax_20.plot([pos1, pos1, pos2, pos2], 
                   [y_line, y_line + y_offset_step * 0.3, 
                    y_line + y_offset_step * 0.3, y_line], 
                   'k-', linewidth=0.8)
        
        # P-value with significance stars
        if p_val < 0.05:
            if p_val < 0.01:
                sig_marker = '**'
            else:
                sig_marker = '*'
            p_text = f'p = {format_pvalue(p_val)} {sig_marker}'
        else:
            p_text = f'p = {format_pvalue(p_val)}'
        
        ax_20.text((pos1 + pos2) / 2, y_line + y_offset_step * 0.4, p_text, 
                   ha='center', va='bottom', fontsize=6.5, color='#000000')
    
    # Add panel labels - A, C, E aligned and shifted left; E aligns with C
    # Panel labels positioned at upper left corner, aligned with y-axis title
    ax_00.text(-0.20, 1.05, 'A', transform=ax_00.transAxes, 
               fontsize=11, va='top', ha='right', color='black')
    ax_01.text(-0.15, 1.05, 'B', transform=ax_01.transAxes, 
               fontsize=11, va='top', ha='right', color='black')
    ax_10.text(-0.20, 1.05, 'C', transform=ax_10.transAxes, 
               fontsize=11, va='top', ha='right', color='black')
    ax_11.text(-0.15, 1.05, 'D', transform=ax_11.transAxes, 
               fontsize=11, va='top', ha='right', color='black')
    # E aligns with C (same x position as C)
    ax_20.text(-0.05, 1.05, 'E', transform=ax_20.transAxes, 
               fontsize=11, va='top', ha='right', color='black')
    
    # Save
    plt.savefig(f'{output_dir}/Figure3.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/Figure3.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics
    stats_data = []
    
    # Age groups
    age_groups = sorted(df['Age_group'].unique())
    for grp in age_groups:
        subset = df[df['Age_group'] == grp]
        stats_data.append({
            'Group': f'Age_{grp}',
            'N': len(subset),
            'Mean_OS': subset['OS_months'].mean(),
            'Median_OS': subset['OS_months'].median()
        })
    
    # Metastases
    for met_val in [0, 1]:
        subset = df[df['Distant_metastases'] == met_val]
        met_label = 'No_metastases' if met_val == 0 else 'Metastases'
        stats_data.append({
            'Group': f'Metastases_{met_label}',
            'N': len(subset),
            'Mean_OS': subset['OS_months'].mean(),
            'Median_OS': subset['OS_months'].median()
        })
    
    # Pathological subtypes
    for subtype in subtype_order:
        subset = df[df['Pathological_subtype'] == subtype]
        if len(subset) > 0:
            stats_data.append({
                'Group': f'Pathological_{subtype}',
                'N': len(subset),
                'Mean_OS': subset['OS_months'].mean(),
                'Median_OS': subset['OS_months'].median()
            })
    
    # Add ALL pairwise comparisons to stats
    if pairwise_results:
        for pos1, pos2, p_val, i, j in pairwise_results:
            label1 = subtype_order[i]
            label2 = subtype_order[j]
            sig_marker = '**' if p_val < 0.01 else ('*' if p_val < 0.05 else '')
            stats_data.append({
                'Group': f'Pairwise_{label1}_vs_{label2}',
                'N': '',
                'Mean_OS': '',
                'Median_OS': f'p={format_pvalue(p_val)} {sig_marker}'.strip()
            })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(f'{table_dir}/Figure3_statistics.csv', index=False)
    
    print(f"Figure 3 saved to {output_dir}/")
    print(f"Statistics saved to {table_dir}/Figure3_statistics.csv")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    set_nature_style()
    
    data_path = 'Data/20251019-R1-data.xlsx'
    df_full = load_full_data(data_path)
    
    print(f"N = {len(df_full)}")
    
    # Generate Figure 1
    print("\nGenerating Figure 1...")
    generate_figure1(df_full, output_dir='Figures', table_dir='Tables')
    print("Figure1.pdf saved")
    print("Figure1.png saved")
    print("Figure1_statistics.csv saved")
    
    # Generate Figure 2
    print("\nGenerating Figure 2...")
    generate_figure2(df_full, output_dir='Figures', table_dir='Tables')
    print("Figure2.pdf saved")
    print("Figure2.png saved")
    print("Figure2_statistics.csv saved")
    
    # Generate Figure 3
    print("\nGenerating Figure 3...")
    df = load_and_preprocess_data(data_path, age_cutoff=28.5)
    generate_figure3(df, output_dir='Figures', table_dir='Tables')
    print("Figure3.pdf saved")
    print("Figure3.png saved")
    print("Figure3_statistics.csv saved")
