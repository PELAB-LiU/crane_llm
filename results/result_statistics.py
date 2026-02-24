import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from llms.config_llms import config

def _load_and_merge_human_validation_data():
    df_final = pd.read_excel("results/results_parsed_detection_and_diagnosis.xlsx", sheet_name="Final_evaluation", engine="openpyxl")
    
    # Read all sheets from the validation file
    validation_file = "results/results_parsed_human_validation_samples_Jose_validated.xlsx"
    all_sheets = pd.read_excel(validation_file, sheet_name=None, engine="openpyxl")
    
    # Start with the instance column from the final dataframe
    df_validation = df_final[['instance']].copy()
    
    # Process each sheet and add columns with crash_detection_{sheet_name} format
    for sheet_name, sheet_df in all_sheets.items():
        if 'instance' in sheet_df.columns:
            # Get only the second column (index 1)
            if len(sheet_df.columns) > 1:
                second_column = sheet_df.columns[1]
                value_columns = [second_column]
            else:
                value_columns = []
            
            # Rename columns to include sheet name
            renamed_columns = {}
            for col in value_columns:
                new_col_name = f"crash_detection_{sheet_name}"
                renamed_columns[col] = new_col_name
            
            # Rename the columns in the sheet dataframe
            sheet_df_renamed = sheet_df.rename(columns=renamed_columns)
            
            # Merge with the main validation dataframe
            df_validation = df_validation.merge(
                sheet_df_renamed, 
                on='instance', 
                how='outer'
            )
    
    # Fill missing values with empty strings
    df_validation = df_validation.fillna('')
    # Focus on columns with the same names between final and validation data
    common_columns = set(df_final.columns).intersection(set(df_validation.columns))
    df_final_filtered = df_final[list(common_columns)]
    df_validation_filtered = df_validation[list(common_columns)]

    # Merge dataframes on instance name
    df_combined = df_final_filtered.merge(df_validation_filtered, on='instance', suffixes=('_human', '_llm'), how='inner')
    
    return df_combined


def _load_and_merge_judge_data(llm_judge_model_name):
    """Load and merge human, LLM, and reversed LLM judgment data"""
    df_final = pd.read_excel("results/results_parsed_human.xlsx", sheet_name="Final_evaluation", engine="openpyxl")
    if config.current_task == "result parsing llm diagnosis only":
        df_llms = pd.read_excel(f"llms/llms_outputs/results_parsed_diagnosis_{llm_judge_model_name}/results_parsed_diagnosis_{llm_judge_model_name}.xlsx", engine="openpyxl")
    else:
        df_llms = pd.read_excel(f"llms/llms_outputs/results_parsed_{llm_judge_model_name}/results_parsed_{llm_judge_model_name}.xlsx", engine="openpyxl")
    # df_llms_reversed = pd.read_excel("llms/llms_outputs/results_parsed_reversed/results_parsed_reversed.xlsx", engine="openpyxl")
    
    # Focus on columns with the same names
    common_columns = set(df_final.columns).intersection(set(df_llms.columns))
    df_final_filtered = df_final[list(common_columns)]
    df_llms_filtered = df_llms[list(common_columns)]
    # df_llms_reversed_filtered = df_llms_reversed[list(common_columns)]

    # Merge dataframes on instance name
    df_combined = df_final_filtered.merge(df_llms_filtered, on='instance', suffixes=('_human', '_llm'), how='inner')
    
    # # Manually rename columns in df_llms_reversed_filtered to ensure suffix is applied
    # df_llms_reversed_renamed = df_llms_reversed_filtered.copy()
    # for col in df_llms_reversed_renamed.columns:
    #     if col != 'instance':
    #         df_llms_reversed_renamed.rename(columns={col: col + '_llm_reversed'}, inplace=True)
    
    # df_combined = df_combined.merge(df_llms_reversed_renamed, on='instance', how='outer')
    
    return df_combined

def _get_model_definitions():
    """Define the models and their column mappings"""
    return {
        'Gemini 2.5': {
            'code_human': 'crash_detection_gemini_code_human',
            'code_llm': 'crash_detection_gemini_code_llm',
            # 'code_llm_reversed': 'crash_detection_gemini_code_llm_reversed',
            'runinfo_human': 'crash_detection_gemini_runinfo_human',
            'runinfo_llm': 'crash_detection_gemini_runinfo_llm',
            # 'runinfo_llm_reversed': 'crash_detection_gemini_runinfo_llm_reversed'
        },
        'Qwen 2.5': {
            'code_human': 'crash_detection_qwen_code_human',
            'code_llm': 'crash_detection_qwen_code_llm',
            # 'code_llm_reversed': 'crash_detection_qwen_code_llm_reversed',
            'runinfo_human': 'crash_detection_qwen_runinfo_human',
            'runinfo_llm': 'crash_detection_qwen_runinfo_llm',
            # 'runinfo_llm_reversed': 'crash_detection_qwen_runinfo_llm_reversed'
        },
        'GPT-5': {
            'code_human': 'crash_detection_gpt5_code_human',
            'code_llm': 'crash_detection_gpt5_code_llm',
            # 'code_llm_reversed': 'crash_detection_gpt5_code_llm_reversed',
            'runinfo_human': 'crash_detection_gpt5_runinfo_human',
            'runinfo_llm': 'crash_detection_gpt5_runinfo_llm',
            # 'runinfo_llm_reversed': 'crash_detection_gpt5_runinfo_llm_reversed'
        }
    }

def _calculate_performance_metrics(df_combined, models, comparison_type='llm'):
    """Calculate performance metrics comparing human vs LLM or human vs LLM_reversed"""
    performance_results = []
    
    # Track totals for overall statistics
    total_code_matches = 0
    total_code_comparisons = 0
    total_runtime_matches = 0
    total_runtime_comparisons = 0
    
    for model_name, columns in models.items():
        # Determine which LLM columns to use
        if comparison_type == 'llm':
            code_llm_col = columns['code_llm']
            runtime_llm_col = columns['runinfo_llm']
        else:  # comparison_type == 'llm_reversed'
            code_llm_col = columns['code_llm_reversed']
            runtime_llm_col = columns['runinfo_llm_reversed']
        
        # Code-only performance
        if (columns['code_human'] in df_combined.columns and 
            code_llm_col in df_combined.columns):
            valid_comparisons = df_combined.dropna(subset=[columns['code_human'], code_llm_col])
            matches_code = (valid_comparisons[columns['code_human']] == valid_comparisons[code_llm_col]).sum()
            total_code = len(valid_comparisons)
            errors_code = total_code - matches_code
            accuracy_code = matches_code / total_code if total_code > 0 else 0
            
            total_code_matches += matches_code
            total_code_comparisons += total_code
        else:
            errors_code = 0
            accuracy_code = 0
            total_code = 0
        
        # With runtime performance
        if (columns['runinfo_human'] in df_combined.columns and 
            runtime_llm_col in df_combined.columns):
            valid_comparisons = df_combined.dropna(subset=[columns['runinfo_human'], runtime_llm_col])
            matches_runtime = (valid_comparisons[columns['runinfo_human']] == valid_comparisons[runtime_llm_col]).sum()
            total_runtime = len(valid_comparisons)
            errors_runtime = total_runtime - matches_runtime
            accuracy_runtime = matches_runtime / total_runtime if total_runtime > 0 else 0
            
            total_runtime_matches += matches_runtime
            total_runtime_comparisons += total_runtime
        else:
            errors_runtime = 0
            accuracy_runtime = 0
            total_runtime = 0
        
        performance_results.append({
            'Model': model_name,
            'Code_Only_Errors': errors_code,
            'Code_Only_Accuracy': accuracy_code,
            'Code_Only_Sample_Size': total_code,
            'With_Runtime_Errors': errors_runtime,
            'With_Runtime_Accuracy': accuracy_runtime,
            'With_Runtime_Sample_Size': total_runtime
        })
    
    # Add total row
    total_code_errors = total_code_comparisons - total_code_matches
    total_runtime_errors = total_runtime_comparisons - total_runtime_matches
    total_code_accuracy = total_code_matches / total_code_comparisons if total_code_comparisons > 0 else 0
    total_runtime_accuracy = total_runtime_matches / total_runtime_comparisons if total_runtime_comparisons > 0 else 0
    
    performance_results.append({
        'Model': 'TOTAL',
        'Code_Only_Errors': total_code_errors,
        'Code_Only_Accuracy': total_code_accuracy,
        'Code_Only_Sample_Size': total_code_comparisons,
        'With_Runtime_Errors': total_runtime_errors,
        'With_Runtime_Accuracy': total_runtime_accuracy,
        'With_Runtime_Sample_Size': total_runtime_comparisons
    })
    
    return performance_results

# def _calculate_llm_consistency(df_combined, models):
#     """Calculate alignment/consistency between LLM and LLM_reversed (combining code and runtime)"""
#     consistency_results = []
    
#     # Track totals for overall statistics
#     total_matches = 0
#     total_comparisons = 0
    
#     for model_name, columns in models.items():
#         model_matches = 0
#         model_comparisons = 0
        
#         # Code consistency
#         if (columns['code_llm'] in df_combined.columns and 
#             columns['code_llm_reversed'] in df_combined.columns):
#             valid_comparisons = df_combined.dropna(subset=[columns['code_llm'], columns['code_llm_reversed']])
#             code_matches = (valid_comparisons[columns['code_llm']] == valid_comparisons[columns['code_llm_reversed']]).sum()
#             code_total = len(valid_comparisons)
#             model_matches += code_matches
#             model_comparisons += code_total
        
#         # Runtime consistency
#         if (columns['runinfo_llm'] in df_combined.columns and 
#             columns['runinfo_llm_reversed'] in df_combined.columns):
#             valid_comparisons = df_combined.dropna(subset=[columns['runinfo_llm'], columns['runinfo_llm_reversed']])
#             runtime_matches = (valid_comparisons[columns['runinfo_llm']] == valid_comparisons[columns['runinfo_llm_reversed']]).sum()
#             runtime_total = len(valid_comparisons)
#             model_matches += runtime_matches
#             model_comparisons += runtime_total
        
#         consistency_rate = model_matches / model_comparisons if model_comparisons > 0 else 0
        
#         consistency_results.append({
#             'Model': model_name,
#             'Consistency_Matches': model_matches,
#             'Consistency_Total': model_comparisons,
#             'Consistency_Rate': consistency_rate
#         })
        
#         total_matches += model_matches
#         total_comparisons += model_comparisons
    
#     # Add total row
#     total_consistency_rate = total_matches / total_comparisons if total_comparisons > 0 else 0
#     consistency_results.append({
#         'Model': 'TOTAL',
#         'Consistency_Matches': total_matches,
#         'Consistency_Total': total_comparisons,
#         'Consistency_Rate': total_consistency_rate
#     })
    
#     return consistency_results

def generate_llm_judge_result_statistics(llm_judge_model_name):
    """
    Generate comprehensive statistics comparing LLM judge performance against human annotations.
    
    Creates three tables:
    1. Human vs LLM performance (code only and with runtime)
    2. Human vs LLM_reversed performance (code only and with runtime)  
    3. LLM consistency (LLM vs LLM_reversed, combining code and runtime)
    """
    # Load and prepare data
    df_combined = _load_and_merge_judge_data(llm_judge_model_name)
    models = _get_model_definitions()
    
    # Calculate performance metrics
    performance_llm = _calculate_performance_metrics(df_combined, models, 'llm')
    # performance_reversed = _calculate_performance_metrics(df_combined, models, 'llm_reversed')
    # consistency_results = _calculate_llm_consistency(df_combined, models)
    
    # Save to Excel file with multiple sheets
    output_file = f"results/llm_judge_statistics_{llm_judge_model_name}.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: Human vs LLM performance
        df_performance_llm = pd.DataFrame(performance_llm)
        df_performance_llm.to_excel(writer, sheet_name='Human_vs_LLM', index=False)
        
        # # Sheet 2: Human vs LLM_reversed performance
        # df_performance_reversed = pd.DataFrame(performance_reversed)
        # df_performance_reversed.to_excel(writer, sheet_name='Human_vs_LLM_Reversed', index=False)
        
        # # Sheet 3: LLM consistency
        # df_consistency = pd.DataFrame(consistency_results)
        # df_consistency.to_excel(writer, sheet_name='LLM_Consistency', index=False)
    
    return output_file


def create_detailed_comparison_excel(llm_judge_model_name):
    """Create a detailed Excel file showing all comparisons between human and LLM judgments"""
    
    # Load and prepare data
    df_combined = _load_and_merge_judge_data(llm_judge_model_name)
    models = _get_model_definitions()
    
    # Create comparison columns based on model definitions
    comparison_cols = {}
    for model_name, columns in models.items():
        # Code-only comparisons
        if 'code_human' in columns and ('code_llm' in columns):
            comparison_cols[f'{model_name}_code'] = (columns['code_human'], columns['code_llm'])
        
        # Runtime info comparisons
        if 'runinfo_human' in columns and ('runinfo_llm' in columns):
            comparison_cols[f'{model_name}_runinfo'] = (columns['runinfo_human'], columns['runinfo_llm'])
        
        # # Reversed LLM comparisons (if available)
        # if 'code_llm_reversed' in columns:
        #     comparison_cols[f'{model_name}_code_reversed'] = (columns['code_human'], columns['code_llm_reversed'])
        # if 'runinfo_llm_reversed' in columns:
        #     comparison_cols[f'{model_name}_runinfo_reversed'] = (columns['runinfo_human'], columns['runinfo_llm_reversed'])
    
    # Create output dataframe
    output_data = {'instance': df_combined['instance']}
    
    # Add comparison columns
    for col_name, (human_col, pred_col) in comparison_cols.items():
        if human_col in df_combined.columns and (pred_col in df_combined.columns):
            output_data[f'{col_name}_human'] = df_combined[human_col]
            output_data[f'{col_name}_pred'] = df_combined[pred_col]
            # Mark mismatches
            output_data[f'{col_name}_match'] = df_combined[human_col] == df_combined[pred_col]
    
    output_df = pd.DataFrame(output_data)
    
    # Save to Excel
    output_file = f"results/llm_judge_detailed_comparisons_{llm_judge_model_name}.xlsx"
    output_df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Detailed comparison saved to {output_file}")


def _load_and_prepare_data(metric_type='crash_detection'):
    """Load and prepare data for analysis"""
    df_final = pd.read_excel("results/results_parsed_detection_and_diagnosis.xlsx", sheet_name="Final_evaluation", engine="openpyxl")
    df_final = df_final.iloc[:223]
    df_label = pd.read_excel("results/benchmark_labels.xlsx", engine="openpyxl")
    df_label.rename(columns={'nb_name': 'instance'}, inplace=True)
    
    if metric_type == 'crash_detection':
        # For crash detection: only _reproduced cases (remove _fixed suffix)
        df_final = df_final[~df_final['instance'].str.endswith('_fixed')]
        # modify instance values by removing _reproduced suffix
        df_final['instance'] = df_final['instance'].str.replace('_reproduced', '', regex=False)
    else:
        # For accuracy: include all instances (both _fixed and _reproduced)
        # Remove both _fixed and _reproduced suffixes to match benchmark labels
        df_final['instance'] = df_final['instance'].str.replace('_fixed', '', regex=False).str.replace('_reproduced', '', regex=False)
    
    # Merge dataframes on instance name
    df_combined = df_final.merge(df_label, on='instance', how='inner')
    print(f"Merged dataframe shape for {metric_type}: {df_combined.shape}")
    
    return df_combined

def _process_libs_cause(df_combined, cause_type):
    """Process Libs-cause data with special grouping"""
    if cause_type == 'Libs-cause':
        # Create a copy to avoid modifying original data
        df_combined = df_combined.copy()
        
        # Replace None/NaN with "NBspecific"
        df_combined[cause_type] = df_combined[cause_type].fillna("NBspecific")
        df_combined.loc[df_combined[cause_type].isin([None, 'None', '']), cause_type] = "NBspecific"
        
        # Group specified libraries as "other"
        libs_to_group = ['lightgbm', 'matplotlib', 'seaborn', 'statsmodels', 'torchvision']
        df_combined.loc[df_combined[cause_type].isin(libs_to_group), cause_type] = "other"
        
        print(f"After grouping, unique {cause_type} values: {sorted(df_combined[cause_type].unique())}")
    
    return df_combined

def _get_label_model_definitions():
    """Get model column definitions for label analysis"""
    return {
        'Gemini 2.5 Flash': {
            'with_runinfo': 'crash_detection_gemini_runinfo',
            'without_runinfo': 'crash_detection_gemini_code'
        },
        'Qwen 2.5 Coder 32B Instruct': {
            'with_runinfo': 'crash_detection_qwen_runinfo',
            'without_runinfo': 'crash_detection_qwen_code'
        },
        'GPT-5': {
            'with_runinfo': 'crash_detection_gpt5_runinfo',
            'without_runinfo': 'crash_detection_gpt5_code'
        },
        # 'PyLint': {
        #     'with_runinfo': 'pylint_crash_detection_code_runinfo',
        #     'without_runinfo': 'pylint_crash_detection_code'
        # },
        # 'PyRight': {
        #     'with_runinfo': 'pyright_crash_detection_code_runinfo',
        #     'without_runinfo': 'pyright_crash_detection_code'
        # }
    }

def _apply_custom_ordering(causes, cause_type):
    """Apply custom ordering for cause types"""
    if cause_type == 'Libs-cause':
        # Custom order: tensorflow/keras, torch, sklearn, numpy, pandas, other, NBspecific
        desired_order = ['tensorflow/keras', 'torch', 'sklearn', 'numpy', 'pandas', 'other', 'NBspecific']
        ordered_causes = []
        for item in desired_order:
            if item in causes:
                ordered_causes.append(item)
        # Add any remaining items not in the desired order
        for item in causes:
            if item not in ordered_causes:
                ordered_causes.append(item)
        return ordered_causes
    
    elif cause_type == 'label_root_cause':
        # Custom order: API misuse, data confusion, NB specific, implementation error, ML model confusion, deprecated API
        # Map "library cause" to "deprecated API"
        cause_mapping = {'library cause': 'deprecated API'}
        mapped_causes = [cause_mapping.get(cause, cause) for cause in causes]
        
        desired_order = ['API misuse', 'data confusion', 'NB specific', 'implementation error', 'ML model confusion', 'deprecated API']
        ordered_causes = []
        for item in desired_order:
            if item in mapped_causes:
                ordered_causes.append(item)
        # Add any remaining items not in the desired order
        for item in mapped_causes:
            if item not in ordered_causes:
                ordered_causes.append(item)
        return ordered_causes
    
    else:
        # Default: return causes as sorted
        return sorted(causes)

def _calculate_rates(df_combined, models, causes, cause_type):
    """Calculate rates for each model and cause"""
    rates_data = {}
    cause_counts = []  # Store counts for each cause
    filtered_causes = []  # Store causes that meet the minimum threshold
    
    # Create reverse mapping for root cause labels
    reverse_mapping = {}
    if cause_type == 'label_root_cause':
        reverse_mapping = {'deprecated API': 'library cause'}
    
    # First pass: determine which causes meet the threshold
    for cause in causes:
        # Map back to original column value if needed
        original_cause = reverse_mapping.get(cause, cause)
        df_cause_group = df_combined[df_combined[cause_type] == original_cause]
        total = len(df_cause_group)
        
        if total >= 5:
            filtered_causes.append(cause)
            cause_counts.append(total)
    
    # Second pass: calculate rates only for filtered causes
    for i, (model_name, columns) in enumerate(models.items()):
        with_rates = []
        without_rates = []
        
        for cause in filtered_causes:
            # Map back to original column value if needed
            original_cause = reverse_mapping.get(cause, cause)
            df_cause_group = df_combined[df_combined[cause_type] == original_cause]
            total = len(df_cause_group)

            with_correct = (df_cause_group[columns['with_runinfo']] == "correct").sum()
            without_correct = (df_cause_group[columns['without_runinfo']] == "correct").sum()
            
            with_rate = with_correct / total
            without_rate = without_correct / total
            
            with_rates.append(with_rate)
            without_rates.append(without_rate)
        
        rates_data[model_name] = {
            'with_rates': with_rates,
            'without_rates': without_rates
        }
    
    return rates_data, cause_counts, filtered_causes

def _create_plot(rates_data, causes, cause_counts, cause_type, metric_type):
    models = list(rates_data.keys())
    
    # Set up the plot
    sns.set_style("whitegrid")

    if cause_type == 'label_root_cause':
        fig_size = (8, 5.5)
    elif metric_type == 'crash_detection' and cause_type == 'Libs-cause':
        fig_size = (12, 5.2) # 5.2
    else:
        fig_size = (12, 5.8)
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Calculate positions for grouped bars
    n_causes = len(causes)
    n_models = len(models)
    bar_width = 0.12
    group_width = n_models * bar_width * 2 + 0.1  # Space for pairs + gap
    
    x = np.arange(n_causes) * (group_width + 0.2)
    
    # Colors for each model (darker for +RT, lighter for -RT)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # For standalone lines, use the main axis
    line_ax = ax
    
    # Line styles and markers for different models
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, model_name in enumerate(models):
        with_rates = np.array(rates_data[model_name]['with_rates'])
        without_rates = np.array(rates_data[model_name]['without_rates'])
        
        # Calculate improvement ratios for both line modes
        improvement_ratios = []
        for with_rate, without_rate in zip(with_rates, without_rates):
            # if without_rate > 0:
            #     # Normalized ratio: (with_rt - without_rt) / without_rt
            #     ratio = (with_rate - without_rate) / without_rate
            # else:
                # When without_rt is 0, use absolute difference
            ratio = with_rate - without_rate
            improvement_ratios.append(ratio)
        
        # Plot line for this model
        line_x = x + (n_models - 1) * bar_width  # Center the line on the group
        line_ax.plot(line_x, improvement_ratios, 
                    color=colors[i], linewidth=2, alpha=0.9,
                    linestyle=line_styles[i], marker=markers[i], markersize=6,
                    label=f"{model_name} Improvement")
        if cause_type == 'label_root_cause':
            line_ax.set_ylim(-0.11, 0.25)
        else:
            line_ax.set_ylim(-0.11, 0.35)
    
    # Configure the line axis
    line_ax.set_ylabel('Runtime Info Improvement', fontsize=20)
    # Add prominent horizontal reference line at y=0
    line_ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2, zorder=1)

    # Customize the plot
    ylabel = 'Crash Detection Rate' if metric_type == 'crash_detection' else 'Accuracy'
    
    # Set x-axis labels with counts
    ax.set_xticks(x + (n_models - 1) * bar_width)
    # Create labels with category name and count
    labels_with_counts = [f"{cause}\n({count})" for cause, count in zip(causes, cause_counts)]
    ax.set_xticklabels(labels_with_counts, rotation=45, ha='right', fontsize=20)
    
    # Set tick label sizes for both axes
    ax.tick_params(axis='y', labelsize=12)
    line_ax.tick_params(axis='y', labelsize=12)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x*100:.0f}'))

    # Create legend patches for models (always needed)
    model_patches = []
    for i, model_name in enumerate(models):
        model_patches.append(mpatches.Patch(color=colors[i], label=model_name))

    improvement_lines = []
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, model_name in enumerate(models):
        improvement_lines.append(
            mlines.Line2D([], [], color=colors[i], linestyle=line_styles[i], 
                            marker=markers[i], linewidth=2, alpha=0.9,
                            label=f"{model_name}")
        )

    if (not (metric_type == 'crash_detection' and cause_type == 'Libs-cause')) and (cause_type != 'label_root_cause'):
        ax.legend(handles=improvement_lines, bbox_to_anchor=(0.5, 1.2), loc='upper center', title='', fontsize=20, frameon=True, ncol=len(improvement_lines))
    # if cause_type == 'label_root_cause':
    #     ax.legend(handles=improvement_lines, bbox_to_anchor=(0.5, 1.3), loc='upper center', title='', fontsize=20, frameon=True, ncol=len(improvement_lines))

    # Adjust layout to accommodate top legends
    plt.subplots_adjust(top=0.8)  # Increase top margin for legends
    plt.tight_layout()
    
    # Save plot
    # Set file name based on parameters
    if cause_type == 'label_root_cause':
        filename_suffix = 'root_cause'
    elif cause_type == 'Libs-cause':
        filename_suffix = 'libs_cause'
    else:
        filename_suffix = cause_type.lower().replace('-', '_').replace(' ', '_')
    if metric_type == 'accuracy':
        filename_suffix += '_accuracy'
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{metric_type}_by_model_and_{filename_suffix}.pdf")
    plt.savefig(output_file) #, bbox_inches='tight'
    print(f"{ylabel} plot saved to {output_file}")
    
    # Show plot immediately
    plt.show()
    plt.close()

def generate_label_results_statistics(cause_type='label_root_cause', metric_type='crash_detection'):
    """Generate statistics for crash detection results against crash labels"""
    
    # Load and prepare data (automatically filters based on metric_type)
    df_combined = _load_and_prepare_data(metric_type)
    
    # Validate cause_type parameter
    if cause_type not in df_combined.columns:
        available_cols = [col for col in df_combined.columns if 'cause' in col.lower() or 'libs' in col.lower()]
        print(f"Available cause columns: {available_cols}")
        raise ValueError(f"Invalid cause_type '{cause_type}'. Available columns: {available_cols}")
    
    # Process special cases for Libs-cause
    df_combined = _process_libs_cause(df_combined, cause_type)
    
    # Get model definitions
    models = _get_label_model_definitions()
    
    # # Filter out Static Analysis tools if requested
    # sa_tools = ['PyLint', 'PyRight']
    # models = {k: v for k, v in models.items() if k not in sa_tools}
    
    # Get unique causes for the specified type
    causes = df_combined[cause_type].unique()
    causes = causes[~pd.isna(causes)]  # Remove NaN values
    
    # Apply custom ordering
    causes = _apply_custom_ordering(causes, cause_type)
    print(f"{cause_type} values: {causes}")
    
    # Calculate rates and get filtered causes
    rates_data, cause_counts, filtered_causes = _calculate_rates(df_combined, models, causes, cause_type)
    
    # Create plot with filtered causes
    filtered_causes = [cause if cause != "tensorflow/keras" else "tensorflow\n/keras" for cause in filtered_causes]
    filtered_causes = [cause if cause != "implementation error" else "implementation\nerror" for cause in filtered_causes]
    _create_plot(rates_data, filtered_causes, cause_counts, cause_type, metric_type)


def calculate_cohens_kappa(llm_judge_model_name=None):
    from sklearn.metrics import cohen_kappa_score
    if llm_judge_model_name is None:
        df_combined = _load_and_merge_human_validation_data()
    else:
        df_combined = _load_and_merge_judge_data(llm_judge_model_name)

    # Define model column pairs for comparison
    models = {
        'crash_detection_gemini_code': ('crash_detection_gemini_code_human', 'crash_detection_gemini_code_llm'),
        'crash_detection_qwen_code': ('crash_detection_qwen_code_human', 'crash_detection_qwen_code_llm'),
        'crash_detection_gpt5_code': ('crash_detection_gpt5_code_human', 'crash_detection_gpt5_code_llm'),
        'crash_detection_gemini_runinfo': ('crash_detection_gemini_runinfo_human', 'crash_detection_gemini_runinfo_llm'),
        'crash_detection_qwen_runinfo': ('crash_detection_qwen_runinfo_human', 'crash_detection_qwen_runinfo_llm'),
        'crash_detection_gpt5_runinfo': ('crash_detection_gpt5_runinfo_human', 'crash_detection_gpt5_runinfo_llm'),
    }
    
    # Collect all human and judge predictions
    all_human_predictions = []
    all_judge_predictions = []
    all_disagreements = []  # Store all disagreement details
    total_disagreement_count = 0
    
    for model_name, (human_col, judge_col) in models.items():
        if human_col in df_combined.columns and judge_col in df_combined.columns:
            # Get valid data for this model (no missing values and no empty strings)
            valid_data = df_combined[['instance', human_col, judge_col]].dropna()
            valid_data = valid_data[(valid_data[human_col] != '') & (valid_data[judge_col] != '')]
            print(f"{model_name}: valid samples for kappa calculation: {len(valid_data)}")
            
            # Check for disagreements and print instance names
            disagreements = valid_data[valid_data[human_col] != valid_data[judge_col]]
            if len(disagreements) > 0:
                print(f"  Disagreements in {model_name} ({len(disagreements)} instances):")
                all_disagreements.append(f"Disagreements in {model_name} ({len(disagreements)} instances):")
                for _, row in disagreements.iterrows():
                    disagreement_detail = f"  {row['instance']}: Human={row[human_col]}, Validator={row[judge_col]}"
                    print(f"    {disagreement_detail}")
                    all_disagreements.append(f"  {disagreement_detail}")
                total_disagreement_count += len(disagreements)
                all_disagreements.append("")  # Add blank line for readability
            
            if len(valid_data) > 0:
                all_human_predictions.extend(valid_data[human_col].tolist())
                all_judge_predictions.extend(valid_data[judge_col].tolist())
    
    # Calculate overall Cohen's kappa
    if len(all_human_predictions) > 0 and len(all_judge_predictions) > 0:
        total_kappa = cohen_kappa_score(all_human_predictions, all_judge_predictions)
        print(f"Overall Cohen's kappa between human and validator results: {total_kappa:.4f}")
        print(f"Total sample size: {len(all_human_predictions)}")
        
        # write to a text file
        if llm_judge_model_name is None:
            llm_judge_model_name = "human_validation"
        with open(f"results/cohens_kappa_{llm_judge_model_name}.txt", "w") as f:
            f.write(f"Overall Cohen's kappa between human and validator results: {total_kappa:.4f}\n")
            f.write(f"Total sample size: {len(all_human_predictions)}\n")
            f.write(f"Total disagreements: {total_disagreement_count}\n\n")
            
            f.write("Detailed disagreements by model:\n")
            f.write("=" * 50 + "\n")
            for line in all_disagreements:
                f.write(line + "\n")
            
            if total_disagreement_count > 0:
                f.write(f"\nFinal Summary:\n")
                f.write(f"Total instances with disagreements: {total_disagreement_count}\n")
                f.write(f"Agreement rate: {(len(all_human_predictions) - total_disagreement_count) / len(all_human_predictions):.4f}\n")
        return total_kappa
    else:
        print("No valid data found for Cohen's kappa calculation")
        return None

def smart_calculate_cohens_kappa(llm_judge_model_name):
    from sklearn.metrics import cohen_kappa_score
    from llms import result_check

    df_combined = _load_and_merge_judge_data(llm_judge_model_name)
  
    # Define model column pairs for comparison
    models = {
        'crash_detection_gemini_code': ('crash_detection_gemini_code_human', 'crash_detection_gemini_code_llm'),
        'crash_detection_qwen_code': ('crash_detection_qwen_code_human', 'crash_detection_qwen_code_llm'),
        'crash_detection_gpt5_code': ('crash_detection_gpt5_code_human', 'crash_detection_gpt5_code_llm'),
        'crash_detection_gemini_runinfo': ('crash_detection_gemini_runinfo_human', 'crash_detection_gemini_runinfo_llm'),
        'crash_detection_qwen_runinfo': ('crash_detection_qwen_runinfo_human', 'crash_detection_qwen_runinfo_llm'),
        'crash_detection_gpt5_runinfo': ('crash_detection_gpt5_runinfo_human', 'crash_detection_gpt5_runinfo_llm'),
    }
    
    # Collect predictions and calculate kappa per model scenario
    per_model_results = {}
    all_human_predictions = []
    all_judge_predictions = []
    
    for model_name, (human_col, judge_col) in models.items():
        if human_col in df_combined.columns and (judge_col in df_combined.columns):
            if len(df_combined) > 0:
                # Calculate per-model kappa - only consider non-empty values
                valid_data = df_combined[[human_col, judge_col]].dropna()
                valid_data = valid_data[(valid_data[human_col] != '') & (valid_data[judge_col] != '')]
                model_human = valid_data[human_col].tolist()
                model_judge = valid_data[judge_col].tolist()
                model_kappa = cohen_kappa_score(model_human, model_judge)
                
                # Count disagreements
                disagreements = sum(1 for h, j in zip(model_human, model_judge) if h != j)
                agreement_rate = (len(model_human) - disagreements) / len(model_human)
                
                per_model_results[model_name] = {
                    'kappa': model_kappa,
                    'total_samples': len(model_human),
                    'disagreements': disagreements,
                    'agreement_rate': agreement_rate
                }
                # print(f"{model_name}: kappa={model_kappa:.4f}, samples={len(model_human)}, disagreements={disagreements}")
                
                # Add to overall calculation
                all_human_predictions.extend(model_human)
                all_judge_predictions.extend(model_judge)
    
    # Calculate overall Cohen's kappa
    if len(all_human_predictions) > 0 and len(all_judge_predictions) > 0:
        total_kappa = cohen_kappa_score(all_human_predictions, all_judge_predictions)
        total_disagreements = sum(1 for h, j in zip(all_human_predictions, all_judge_predictions) if h != j)
        
        print(f"\n{llm_judge_model_name} Overall Results:")
        print(f"Overall Cohen's kappa: {total_kappa:.4f}")
        print(f"Total sample size: {len(all_human_predictions)}")
        print(f"Total disagreements: {total_disagreements}")
        print(f"Overall agreement rate: {(len(all_human_predictions) - total_disagreements) / len(all_human_predictions):.4f}")
        
        # Write detailed results to file
        with open(f"results/cohens_kappa_smart_{llm_judge_model_name}.txt", "w") as f:
            f.write(f"Overall Cohen's kappa between LLM judge and human results: {total_kappa:.4f}\n")
            f.write(f"Total sample size: {len(all_human_predictions)}\n")
            f.write(f"Total disagreements: {total_disagreements}\n")
            f.write(f"Overall agreement rate: {(len(all_human_predictions) - total_disagreements) / len(all_human_predictions):.4f}\n\n")
            
            f.write("Per-model breakdown:\n")
            for model_name, results in per_model_results.items():
                f.write(f"{model_name}: kappa={results['kappa']:.4f}, samples={results['total_samples']}, disagreements={results['disagreements']}, agreement_rate={results['agreement_rate']:.4f}\n")
        
        return total_kappa
    else:
        print("No valid data found for Cohen's kappa calculation")
        return None