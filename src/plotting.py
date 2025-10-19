"""Plotting functions for survival curves and clinical figures."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from lifelines import KaplanMeierFitter
from scipy import stats

from .config import COLORS
from .statistics import (
    compute_logrank_test, compute_cox_hr, 
    compute_median_survival, compute_multivariate_logrank
)


def format_pvalue(p):
    """Format p-value with appropriate precision."""
    if p < 0.001:
        return f"{p:.2e}"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def add_number_at_risk(ax, kmf_dict, colors_dict, xlabel='Time (months)', 
                       time_points=None, y_offset=-0.10):
    """
    Add number at risk table below KM plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    kmf_dict : dict
        Group labels to fitted KaplanMeierFitter objects
    colors_dict : dict
        Group labels to colors
    xlabel : str
        X-axis label
    time_points : array-like, optional
    y_offset : float
        Vertical position offset
    """
    if time_points is None:
        all_times = []
        for kmf in kmf_dict.values():
            all_times.extend(kmf.event_table.index.tolist())
        max_time = max(all_times) if all_times else 100
        time_points = np.linspace(0, max_time, 5).astype(int)
    
    xlim = ax.get_xlim()
    ax.set_xticks(time_points)
    ax.set_xticklabels([])
    
    n_risk_y_center = y_offset - (len(kmf_dict) - 1) * 0.06 / 2
    ax.text(-0.12, n_risk_y_center - 0.01, 'N@risk',
            transform=ax.transAxes, fontsize=8, ha='center', 
            va='center', rotation=90, color='#000000')
    
    for i, (label, kmf) in enumerate(kmf_dict.items()):
        y_pos = y_offset - i * 0.06
        
        line_color = None
        for line in ax.get_lines():
            if line.get_label() == label:
                line_color = line.get_color()
                break
        if line_color is None:
            line_color = colors_dict.get(label, COLORS['blue'])
        
        ax.add_patch(plt.Rectangle((-0.085, y_pos - 0.018), 0.045, 0.036,
                                   transform=ax.transAxes, facecolor=line_color,
                                   edgecolor='none', clip_on=False))
        
        for t in time_points:
            n_at_risk = (kmf.event_table.loc[kmf.event_table.index <= t, 'at_risk']
                        .iloc[-1] if len(kmf.event_table.loc[kmf.event_table.index <= t]) > 0 else 0)
            x_norm = (t - xlim[0]) / (xlim[1] - xlim[0])
            ax.text(x_norm, y_pos, str(int(n_at_risk)),
                   transform=ax.transAxes, fontsize=9.5, ha='center',
                   va='center', color='#000000', weight='bold')
    
    xlabel_y = y_offset - len(kmf_dict) * 0.06 - 0.08
    ax.text(0.5, xlabel_y, xlabel, transform=ax.transAxes,
           fontsize=8, ha='center', va='top', color='#000000')


def plot_km_curve_enhanced(ax, df, time_col, event_col, group_col=None,
                          show_hr=True, show_median=True, show_n_at_risk=True,
                          xlabel='Time (months)', title='', min_n=1,
                          reference_group=None):
    """
    Plot enhanced Kaplan-Meier survival curves with statistics.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df : pd.DataFrame
    time_col : str
    event_col : str
    group_col : str, optional
    show_hr : bool
    show_median : bool
    show_n_at_risk : bool
    xlabel : str
    title : str
    min_n : int
    reference_group : str, optional
    
    Returns
    -------
    dict
        Statistics dictionary
    """
    kmf = KaplanMeierFitter()
    stats_dict = {}
    kmf_dict = {}
    colors_dict = {}
    
    df_clean = df[[time_col, event_col]].copy()
    if group_col:
        df_clean[group_col] = df[group_col]
    df_clean = df_clean.dropna()
    
    if group_col is None or df_clean[group_col].nunique() == 1:
        kmf.fit(df_clean[time_col], event_observed=df_clean[event_col], label='All patients')
        kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=1.5, color=COLORS['blue'])
        kmf_dict['All patients'] = kmf
        colors_dict['All patients'] = COLORS['blue']
        
        median_surv = compute_median_survival(kmf)
        stats_dict['All_patients'] = {
            'n': len(df_clean),
            'events': df_clean[event_col].sum(),
            'median_survival': median_surv
        }
        
        if show_median and not np.isnan(median_surv):
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.axvline(x=median_surv, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, ymax=0.5)
            ax.text(median_surv, 0.05, f'{median_surv:.2f}',
                   fontsize=7, ha='center', color='#000000')
    else:
        groups = df_clean[group_col].unique()
        groups = [g for g in groups if df_clean[df_clean[group_col] == g].shape[0] >= min_n]
        
        colors = [COLORS['blue'], COLORS['gray']]
        
        medians = []
        for idx, group in enumerate(groups):
            mask = df_clean[group_col] == group
            n = mask.sum()
            label = f'{group}'
            color = colors[idx % len(colors)]
            
            kmf_temp = KaplanMeierFitter()
            kmf_temp.fit(df_clean[mask][time_col], 
                        event_observed=df_clean[mask][event_col], 
                        label=label)
            kmf_temp.plot_survival_function(ax=ax, ci_show=False, 
                                           linewidth=1.5, color=color,
                                           at_risk_counts=False)
            
            kmf_dict[label] = kmf_temp
            colors_dict[label] = color
            
            median_surv = compute_median_survival(kmf_temp)
            medians.append(median_surv)
            
            stats_dict[f'{group}'] = {
                'n': n,
                'events': df_clean[mask][event_col].sum(),
                'median_survival': median_surv
            }
        
        if len(groups) == 2:
            try:
                p_val = compute_logrank_test(df_clean, time_col, event_col, group_col, groups[0], groups[1])
                stats_dict['p_value'] = p_val
                
                hr_result = compute_cox_hr(df_clean, time_col, event_col, group_col, reference_group)
                if hr_result:
                    stats_dict['hr_results'] = hr_result
            except:
                pass
        elif len(groups) > 2:
            try:
                p_val = compute_multivariate_logrank(df_clean, time_col, event_col, group_col)
                stats_dict['p_value'] = p_val
            except:
                pass
        
        if show_median:
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            
            medians_valid = [(m, i) for i, m in enumerate(medians) if not np.isnan(m)]
            if medians_valid:
                medians_sorted = sorted(medians_valid, key=lambda x: x[0])
                
                for m, idx in medians_valid:
                    ax.axvline(x=m, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, ymax=0.5)
                    
                    if m == medians_sorted[0][0]:
                        ha = 'right'
                    elif m == medians_sorted[-1][0]:
                        ha = 'left'
                    else:
                        ha = 'center'
                    
                    ax.text(m, 0.05, f'{m:.2f}',
                           fontsize=7, ha=ha, color='#000000')
        
        if 'p_value' in stats_dict:
            p_text = f"p={format_pvalue(stats_dict['p_value'])}"
            ax.text(0.05, 0.2, p_text, transform=ax.transAxes,
                   fontsize=7, ha='left', va='top', color='#000000')
        
        if show_hr and 'hr_results' in stats_dict:
            hr_res = stats_dict['hr_results']
            hr_text = f"HR {hr_res['hr']:.2f} ({hr_res['ci_lower']:.2f}-{hr_res['ci_upper']:.2f})"
            
            narrow_font = FontProperties(family='DejaVu Sans', stretch='condensed')
            
            title_y = 1.12
            hr_y = title_y - 0.10
            
            ax.text(1.05, title_y, title, transform=ax.transAxes,
                   fontsize=8, ha='left', va='top', color='#000000')
            ax.text(1.05, hr_y, hr_text, transform=ax.transAxes,
                   fontsize=7, ha='left', va='top', color='#000000',
                   fontproperties=narrow_font)
            
            legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.05),
                             frameon=False, fontsize=7,
                             handlelength=1.2, handletextpad=0.4,
                             ncol=1, columnspacing=1.5)
            for text in legend.get_texts():
                text.set_fontproperties(narrow_font)
        else:
            if title:
                narrow_font = FontProperties(family='DejaVu Sans', stretch='condensed')
                ax.text(1.05, 1.12, title, transform=ax.transAxes,
                       fontsize=8, ha='left', va='top', color='#000000')
            
            ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.05),
                     frameon=False, fontsize=7,
                     handlelength=1.2, handletextpad=0.4)
    
    if show_n_at_risk:
        add_number_at_risk(ax, kmf_dict, colors_dict, xlabel=xlabel)
    
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0', '25', '50', '75', '100'], fontsize=7, color='#000000')
    ax.set_xlabel('')
    ax.set_ylabel('Survival (%)', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return stats_dict


def plot_boxplot_with_points(ax, df, y_col, group_col, group_order=None, colors_dict=None):
    """
    Plot boxplot with scatter points.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df : pd.DataFrame
    y_col : str
    group_col : str
    group_order : list, optional
    colors_dict : dict, optional
    
    Returns
    -------
    tuple
        (data_list, labels_list, positions_list)
    """
    if group_order is None:
        group_order = sorted(df[group_col].unique())
    
    data_list = []
    labels_list = []
    positions_list = []
    
    for i, group in enumerate(group_order):
        subset = df[df[group_col] == group][y_col]
        if len(subset) > 0:
            data_list.append(subset)
            labels_list.append(group)
            positions_list.append(i)
    
    bp = ax.boxplot(data_list, positions=positions_list, widths=0.5,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='none', edgecolor='#000000', linewidth=0.8),
                    whiskerprops=dict(color='#000000', linewidth=0.8),
                    capprops=dict(color='#000000', linewidth=0.8),
                    medianprops=dict(color='#000000', linewidth=1.2))
    
    for i, data in enumerate(data_list):
        y = data.values
        x = np.random.normal(positions_list[i], 0.08, size=len(y))
        color = colors_dict.get(labels_list[i], COLORS['blue']) if colors_dict else COLORS['blue']
        ax.scatter(x, y, alpha=0.7, s=25, color=color,
                  edgecolors='white', linewidth=0.5, zorder=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=7, colors='#000000')
    
    return data_list, labels_list, positions_list


def plot_scatter_age_os(ax, df, age_col='Age', os_col='OS_months', age_cutoff=28.5):
    """
    Plot age vs OS scatter plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df : pd.DataFrame
    age_col : str
    os_col : str
    age_cutoff : float
    """
    df_plot = df[[age_col, os_col]].dropna()
    
    colors = df_plot[age_col].apply(
        lambda x: COLORS['blue'] if x <= age_cutoff else COLORS['gray']
    )
    
    ax.scatter(df_plot[age_col], df_plot[os_col], c=colors, s=30,
              alpha=0.7, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Age (years)', fontsize=8, color='#000000')
    ax.set_ylabel('OS (months)', fontsize=8, color='#000000')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=7, colors='#000000')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['blue'],
               markersize=5, label=f'≤{age_cutoff}', markeredgewidth=0.5, markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['gray'],
               markersize=5, label=f'>{age_cutoff}', markeredgewidth=0.5, markeredgecolor='white')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=7,
             ncol=2, columnspacing=1.5, handletextpad=0.3)

