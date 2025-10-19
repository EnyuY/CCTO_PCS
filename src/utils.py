"""Utility functions."""

import pandas as pd
import os


def format_pvalue(p):
    """Format p-value with appropriate precision."""
    if p < 0.001:
        return f"{p:.2e}"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def merge_statistics_to_excel(table_dir='Tables', output_filename='supplemental_table_1.xlsx'):
    """
    Merge all CSV files in table directory into a single Excel file.
    
    Parameters
    ----------
    table_dir : str
        Directory containing CSV files
    output_filename : str
        Output Excel filename
    """
    output_filepath = os.path.join(table_dir, output_filename)
    csv_files = sorted([f for f in os.listdir(table_dir) if f.endswith('.csv')])
    
    if not csv_files:
        print("No CSV files found.")
        return
    
    with pd.ExcelWriter(output_filepath, engine='openpyxl') as writer:
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(table_dir, csv_file))
            sheet_name = os.path.splitext(csv_file)[0]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Supplemental table: {output_filepath} ({len(csv_files)} sheets)")

