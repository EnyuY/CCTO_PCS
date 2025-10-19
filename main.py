#!/usr/bin/env python3
"""
Main script for generating all figures and tables.
Simplified entry point that calls modular functions.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import set_nature_style
from src.data_loader import load_full_data, load_and_preprocess_data  
from src.utils import merge_statistics_to_excel

# Import from existing FiguresOuter for now (will modularize later)
from FiguresOuter import generate_figure1, generate_figure2, generate_figure3


def main():
    """Main analysis workflow."""
    set_nature_style()
    
    data_path = 'Data/20251019-R1-data.xlsx'
    df_full = load_full_data(data_path)
    
    print(f"N = {len(df_full)}")
    
    # Generate Figure 1
    print("\nGenerating Figure 1...")
    generate_figure1(df_full)
    print("Figure 1 completed")
    
    # Generate Figure 2
    print("\nGenerating Figure 2...")
    generate_figure2(df_full)
    print("Figure 2 completed")
    
    # Generate Figure 3
    print("\nGenerating Figure 3...")
    df = load_and_preprocess_data(data_path)
    generate_figure3(df)
    print("Figure 3 completed")
    
    # Merge statistics
    print("\nMerging statistics...")
    merge_statistics_to_excel()
    
    print("\n✓ All figures and tables generated")


if __name__ == '__main__':
    main()

