"""
Manual Analysis Sampling Module

This module provides functionality for stratified sampling of data for manual analysis.
It performs statistical sampling with configurable confidence levels and margin of error,
and creates Excel outputs with multiple sheets for each prediction column.
"""

import pandas as pd
import numpy as np
import math
from pathlib import Path


def calculate_sample_size(population_size, confidence_level=0.95, margin_error=0.05):
    """Calculate sample size for statistical sampling"""
    # Z-score for 95% confidence level
    z_score = 1.96 if confidence_level == 0.95 else 1.645
    
    # Formula: n = (Z^2 * p * (1-p)) / E^2
    # Using p = 0.5 for maximum variability (worst case)
    p = 0.5
    numerator = (z_score ** 2) * p * (1 - p)
    denominator = margin_error ** 2
    
    # Sample size for infinite population
    n_infinite = numerator / denominator
    
    # Adjust for finite population
    n_adjusted = n_infinite / (1 + ((n_infinite - 1) / population_size))
    
    return math.ceil(n_adjusted)


def get_prediction_columns(df):
    """Get all prediction columns (exclude instance column)"""
    return [col for col in df.columns if col != 'instance']


def calculate_non_empty_counts(df, prediction_cols):
    """Calculate non-empty value counts for each prediction column"""
    non_empty_counts = {}
    for col in prediction_cols:
        non_empty_counts[col] = df[col].notna().sum()
    return non_empty_counts


def calculate_proportional_sample_sizes(non_empty_counts, total_sample_size):
    """Calculate proportional sample sizes based on non-empty counts"""
    total_non_empty = sum(non_empty_counts.values())
    column_sample_sizes = {}
    
    for col, count in non_empty_counts.items():
        proportion = count / total_non_empty
        col_sample_size = max(1, math.ceil(total_sample_size * proportion))
        column_sample_sizes[col] = col_sample_size
    
    return column_sample_sizes


def extract_library_and_number(instance_name):
    """Extract library name and number from instance name"""
    # Remove '_reproduced' suffix if present
    clean_name = instance_name.replace('_reproduced', '')
    
    # Split by underscore and get library name and number
    parts = clean_name.split('_')
    if len(parts) >= 2:
        library = parts[0]
        try:
            number = int(parts[1])
        except ValueError:
            number = 0
    else:
        library = clean_name
        number = 0
    
    return library, number

def sort_instances(df):
    """Sort instances by library order then by number"""
    # Define library order
    library_order = ['tensorflow', 'torch', 'sklearn', 'pandas', 'numpy', 
                    'NBspecific', 'matplotlib', 'seaborn', 'statsmodels', 
                    'lightgbm', 'torchvision']
    
    # Extract library and number for sorting
    df_copy = df.copy()
    df_copy[['library', 'number']] = df_copy['instance'].apply(
        lambda x: pd.Series(extract_library_and_number(x))
    )
    
    # Create library order mapping
    library_order_map = {lib: idx for idx, lib in enumerate(library_order)}
    df_copy['library_order'] = df_copy['library'].map(library_order_map)
    
    # Sort by library order, then by number
    df_sorted = df_copy.sort_values(['library_order', 'number'], na_position='last')
    
    # Return only instance column
    return df_sorted[['instance']].reset_index(drop=True)

def sample_column_data(df, col, sample_size):
    """Sample data for a specific column - return only instance column for human evaluation"""
    col_data = df[df[col].notna()]
    if len(col_data) == 0:
        return pd.DataFrame(columns=['instance'])
    
    if len(col_data) <= sample_size:
        sampled_indices = col_data.index.tolist()
    else:
        sampled_indices = np.random.choice(
            col_data.index, 
            size=sample_size, 
            replace=False
        ).tolist()
    
    sampled_df = df.loc[sampled_indices, ['instance']].reset_index(drop=True)
    
    # Sort the sampled data
    return sort_instances(sampled_df)


def print_sampling_allocation(non_empty_counts, column_sample_sizes, total_sample_size):
    """Print detailed sampling allocation information"""
    total_non_empty = sum(non_empty_counts.values())

    print("-" * 60)
    
    for col in column_sample_sizes.keys():
        proportion = non_empty_counts[col] / total_non_empty
        print(f"{col}:")
        print(f"  Non-empty values: {non_empty_counts[col]}")
        print(f"  Proportion: {proportion:.3f}")
        print(f"  Sample size: {column_sample_sizes[col]}")
        print()


def stratified_sampling_by_column_coverage(df, total_sample_size):
    """
    Perform stratified sampling based on non-empty value counts per column
    """
    prediction_cols = get_prediction_columns(df)
    non_empty_counts = calculate_non_empty_counts(df, prediction_cols)
    column_sample_sizes = calculate_proportional_sample_sizes(non_empty_counts, total_sample_size)
    
    print_sampling_allocation(non_empty_counts, column_sample_sizes, total_sample_size)
    
    # Sample data for each column
    sampled_data = {}
    for col in prediction_cols:
        sampled_data[col] = sample_column_data(df, col, column_sample_sizes[col])
    
    return sampled_data, column_sample_sizes


def clean_sheet_name(sheet_name):
    """Clean sheet name to comply with Excel restrictions"""
    replacements = {'/': '_', '\\': '_', '*': '_', '?': '_', '[': '_', ']': '_', ':': '_'}
    for old_char, new_char in replacements.items():
        sheet_name = sheet_name.replace(old_char, new_char)
    return sheet_name[:31]  # Excel sheet name limit


def get_clean_sheet_name(col):
    """Convert column names to clean sheet names"""
    sheet_name_mapping = {
        'Gemini 2.5_code_pred': 'gemini_code',
        'Gemini 2.5_runinfo_pred': 'gemini_runinfo', 
        'Qwen 2.5_code_pred': 'qwen_code',
        'Qwen 2.5_runinfo_pred': 'qwen_runinfo',
        'GPT-5_code_pred': 'gpt5_code',
        'GPT-5_runinfo_pred': 'gpt5_runinfo'
    }
    return sheet_name_mapping.get(col, col.replace(' ', '_').replace('.', '_').replace('-', '_'))

def save_to_excel(sampled_data, column_sample_sizes, df, output_path):
    """Save sampled data to Excel with multiple sheets"""
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Individual column sheets
        for col, data in sampled_data.items():
            sheet_name = get_clean_sheet_name(col)
            data.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Sheet '{sheet_name}': {len(data)} samples")


def print_sampling_summary(column_sample_sizes, df):
    """Print final sampling summary"""
    print("\nSampling summary:")
    for col, size in column_sample_sizes.items():
        non_empty_count = df[col].notna().sum()
        coverage = (size / non_empty_count) * 100 if non_empty_count > 0 else 0
        print(f"  {col}: {size} samples ({coverage:.1f}% coverage)")


def load_and_describe_data(file_path):
    """Load data and print basic information"""
    # print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    
    # print(f"Dataset shape: {df.shape}")
    # print(f"Columns: {df.columns.tolist()}")
    
    return df


def run_manual_analysis_sampling(input_file='results/results_parsed_human_population.xlsx', 
                                output_file='results/results_parsed_manual_analysis_samples_Jose.xlsx',
                                confidence_level=0.95, 
                                margin_error=0.05,
                                random_seed=42):
    """
    Main function to orchestrate the manual analysis sampling process
    
    Args:
        input_file (str): Path to the input Excel file
        output_file (str): Path to the output Excel file
        confidence_level (float): Confidence level for sample size calculation
        margin_error (float): Margin of error for sample size calculation
        random_seed (int): Random seed for reproducible sampling
    """
    # Load data
    df = load_and_describe_data(input_file)
    
    # Calculate population size as total non-empty values across all prediction columns
    prediction_cols = get_prediction_columns(df)
    non_empty_counts = calculate_non_empty_counts(df, prediction_cols)
    population_size = sum(non_empty_counts.values())
    overall_sample_size = calculate_sample_size(population_size, confidence_level, margin_error)
    
    print(f"\nPopulation size (total non-empty values): {population_size}")
    print(f"Calculated sample size ({confidence_level*100:.0f}% confidence, {margin_error*100:.0f}% margin of error): {overall_sample_size}")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Perform stratified sampling
    sampled_data, column_sample_sizes = stratified_sampling_by_column_coverage(df, overall_sample_size)
    
    # Save results to Excel
    output_path = Path(output_file)
    save_to_excel(sampled_data, column_sample_sizes, df, output_path)
    
    # Print summaries
    print_sampling_summary(column_sample_sizes, df)
    
    print(f"\nSampling complete! Results saved to {output_path}")
    return sampled_data, column_sample_sizes

def filter_files_by_sample(path_sample_file='results/results_parsed_manual_analysis_samples_Jose.xlsx', 
                           path_all_results='llms/llms_outputs/results_raw/', 
                           path_output_filtered_results='results/results_raw_samples_Jose/'):
    """
    Filter and copy files from source directories based on sampled instances.
    
    Args:
        path_sample_file (str): Path to the Excel file containing sampled instances
        path_all_results (str): Path to the directory containing all result folders
        path_output_filtered_results (str): Path to output directory for filtered results
    """
    import os
    import shutil
    
    # Convert paths to Path objects for easier handling
    sample_file_path = Path(path_sample_file)
    all_results_path = Path(path_all_results)
    output_path = Path(path_output_filtered_results)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read the Excel file to get all sheet names and their instances
    excel_file = pd.ExcelFile(sample_file_path)
    sheet_names = excel_file.sheet_names
    
    print(f"Processing {len(sheet_names)} sheets from {sample_file_path}")
    
    for sheet_name in sheet_names:
        print(f"\nProcessing sheet: {sheet_name}")
        
        # Read the sheet to get instances
        df_sheet = pd.read_excel(sample_file_path, sheet_name=sheet_name)
        
        if 'instance' not in df_sheet.columns:
            print(f"  Warning: No 'instance' column found in sheet {sheet_name}")
            continue
        
        # Get list of instances from the sheet
        instances = df_sheet['instance'].dropna().tolist()
        print(f"  Found {len(instances)} instances to process")
        
        # Construct source folder path
        source_folder = all_results_path / f"crash_detection_{sheet_name}"
        
        if not source_folder.exists():
            print(f"  Warning: Source folder {source_folder} does not exist")
            continue
        
        # Create output subfolder for this sheet
        output_subfolder = output_path / sheet_name
        output_subfolder.mkdir(parents=True, exist_ok=True)
        
        # Copy files for each instance
        copied_count = 0
        for instance in instances:
            # Look for files that match this instance name
            source_files = list(source_folder.glob(f"{instance}*"))
            
            if not source_files:
                print(f"    Warning: No files found for instance {instance}")
                continue
            
            # Copy all matching files
            for source_file in source_files:
                destination_file = output_subfolder / source_file.name
                try:
                    shutil.copy2(source_file, destination_file)
                    copied_count += 1
                except Exception as e:
                    print(f"    Error copying {source_file.name}: {e}")
        
        print(f"  Copied {copied_count} files to {output_subfolder}")
    
    print(f"\nFiltering complete! Results saved to {output_path}")
    return True
