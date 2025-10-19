"""Configuration and style settings."""

import matplotlib.pyplot as plt


def set_nature_style():
    """Apply Nature Medicine journal style parameters."""
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


# Color schemes
COLORS = {
    'blue': '#0173B2',
    'gray': '#999999',
    'black': '#000000'
}

