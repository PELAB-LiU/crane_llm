import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def generate_llm_judge_result_statistics(parsed_summary_sa_filename, parsed_summary_llm_filename, output_filename):
    df_human = pd.read_excel("results/results_human_evaluated.xlsx", sheet_name="Human_evaluation", engine="openpyxl")
    df_sas = pd.read_excel(f"sas/sas_outputs/{parsed_summary_sa_filename}.xlsx", engine="openpyxl")
    df_llms = pd.read_excel(f"llms/llms_outputs/{parsed_summary_llm_filename}.xlsx", engine="openpyxl")
    
    # Merge dataframes on instance name
    df_combined = df_human.merge(df_llms, on='instance', suffixes=('_human', '_llm'), how='outer')
    df_combined = df_combined.merge(df_sas, on='instance', suffixes=('', '_sas'), how='outer')
    
    # Define the models and their columns
    models = {
        'gemini 2.5 flash': {
            'code_human': 'gemini_2_5_flash_crash_detection_code_human',
            'code_pred': 'gemini_2_5_flash_crash_detection_code_llm',
            'runinfo_human': 'gemini_2_5_flash_crash_detection_code_runinfo_human',
            'runinfo_pred': 'gemini_2_5_flash_crash_detection_code_runinfo_llm'
        },
        'qwen 2.5 32b instruct': {
            'code_human': 'Qwen_2_5_32B_Instruct_crash_detection_code_human',
            'code_pred': 'Qwen_2_5_32B_Instruct_crash_detection_code_llm',
            'runinfo_human': 'Qwen_2_5_32B_Instruct_crash_detection_code_runinfo_human',
            'runinfo_pred': 'Qwen_2_5_32B_Instruct_crash_detection_code_runinfo_llm'
        },
        'gpt 5': {
            'code_human': 'gpt_5_crash_detection_code_human',
            'code_pred': 'gpt_5_crash_detection_code_llm',
            'runinfo_human': 'gpt_5_crash_detection_code_runinfo_human',
            'runinfo_pred': 'gpt_5_crash_detection_code_runinfo_llm'
        },
        'pylint': {
            'code_human': 'pylint_crash_detection_code',
            'code_pred': 'pylint_crash_detection_code_sas',
            'runinfo_human': 'pylint_crash_detection_code_runinfo',
            'runinfo_pred': 'pylint_crash_detection_code_runinfo_sas'
        },
        'pyright': {
            'code_human': 'pyright_crash_detection_code',
            'code_pred': 'pyright_crash_detection_code_sas',
            'runinfo_human': 'pyright_crash_detection_code_runinfo',
            'runinfo_pred': 'pyright_crash_detection_code_runinfo_sas'
        }
    }
    
    results = []
    
    for model_name, columns in models.items():
        # For -RT (code only)
        human_col = columns['code_human']
        pred_col = columns['code_pred']
        
        # Check if columns exist in the dataframe
        if human_col not in df_combined.columns or pred_col not in df_combined.columns:
            f_code = 0
            accuracy_code = 0
        else:
            # Count mismatches for code-only
            valid_comparisons = df_combined.dropna(subset=[human_col, pred_col])
            f_code = (valid_comparisons[human_col] != valid_comparisons[pred_col]).sum()
            total_code = len(valid_comparisons)
            accuracy_code = (total_code - f_code) / total_code if total_code > 0 else 0
        
        # For +RT (code + runinfo)
        if (columns['runinfo_human'] in df_combined.columns and 
            columns['runinfo_pred'] in df_combined.columns):
            human_runinfo_col = columns['runinfo_human']
            pred_runinfo_col = columns['runinfo_pred']
            
            valid_comparisons_runinfo = df_combined.dropna(subset=[human_runinfo_col, pred_runinfo_col])
            f_runinfo = (valid_comparisons_runinfo[human_runinfo_col] != valid_comparisons_runinfo[pred_runinfo_col]).sum()
            total_runinfo = len(valid_comparisons_runinfo)
            accuracy_runinfo = (total_runinfo - f_runinfo) / total_runinfo if total_runinfo > 0 else 0
        else:
            f_runinfo = '-'
            accuracy_runinfo = '-'
        
        results.append({
            'Model': model_name,
            'F_Minus_RT': f_code,
            'Accuracy_Minus_RT': accuracy_code,
            'F_Plus_RT': f_runinfo,
            'Accuracy_Plus_RT': accuracy_runinfo
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Rename columns for better readability in Excel
    results_df.rename(columns={
        'Model': 'LLM as a Judge',
        'F_Minus_RT': 'F (-RT)',
        'Accuracy_Minus_RT': 'Accuracy (-RT)',
        'F_Plus_RT': 'F (+RT)',
        'Accuracy_Plus_RT': 'Accuracy (+RT)'
    }, inplace=True)
    
    # Save to Excel file
    output_file = f"results/{output_filename}.xlsx"
    results_df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Results saved to {output_file}")
    
    # Also print the table for immediate viewing
    print("\nLLM as a Judge\t|\t-RT\t\t|\t+RT")
    print("\t\t|\tF\t|\taccuracy\t|\tF\t|\taccuracy")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        if row['F (+RT)'] == '-':
            print(f"{row['LLM as a Judge']:<20}\t|\t{row['F (-RT)']}\t|\t{row['Accuracy (-RT)']:.9f}\t|\t{row['F (+RT)']}\t|\t{row['Accuracy (+RT)']}")
        else:
            print(f"{row['LLM as a Judge']:<20}\t|\t{row['F (-RT)']}\t|\t{row['Accuracy (-RT)']:.9f}\t|\t{row['F (+RT)']}\t|\t{row['Accuracy (+RT)']:.9f}")


def create_detailed_comparison_excel(parsed_summary_sa_filename, parsed_summary_llm_filename, output_filename):
    """Create a detailed Excel file showing all comparisons with mismatches highlighted in blue"""
    
    df_human = pd.read_excel("results/results_human_evaluated.xlsx", sheet_name="Human_evaluation", engine="openpyxl")
    df_sas = pd.read_excel(f"sas/sas_outputs/{parsed_summary_sa_filename}.xlsx", engine="openpyxl")
    df_llms = pd.read_excel(f"llms/llms_outputs/{parsed_summary_llm_filename}.xlsx", engine="openpyxl")

    # Merge dataframes
    df_combined = df_human.merge(df_llms, on='instance', suffixes=('_human', '_llm'), how='outer')
    df_combined = df_combined.merge(df_sas, on='instance', suffixes=('', '_sas'), how='outer')
    
    # Define comparison columns
    comparison_cols = {
        'gemini_2_5_flash_code': ('gemini_2_5_flash_crash_detection_code_human', 'gemini_2_5_flash_crash_detection_code_llm'),
        'gemini_2_5_flash_runinfo': ('gemini_2_5_flash_crash_detection_code_runinfo_human', 'gemini_2_5_flash_crash_detection_code_runinfo_llm'),
        'qwen_2_5_32b_code': ('Qwen_2_5_32B_Instruct_crash_detection_code_human', 'Qwen_2_5_32B_Instruct_crash_detection_code_llm'),
        'qwen_2_5_32b_runinfo': ('Qwen_2_5_32B_Instruct_crash_detection_code_runinfo_human', 'Qwen_2_5_32B_Instruct_crash_detection_code_runinfo_llm'),
        'gpt_5_code': ('gpt_5_crash_detection_code_human', 'gpt_5_crash_detection_code_llm'),
        'gpt_5_runinfo': ('gpt_5_crash_detection_code_runinfo_human', 'gpt_5_crash_detection_code_runinfo_llm'),
        'pylint_code': ('pylint_crash_detection_code', 'pylint_crash_detection_code_sas'),
        'pylint_runinfo': ('pylint_crash_detection_code_runinfo', 'pylint_crash_detection_code_runinfo_sas'),
        'pyright_code': ('pyright_crash_detection_code', 'pyright_crash_detection_code_sas'),
        'pyright_runinfo': ('pyright_crash_detection_code_runinfo', 'pyright_crash_detection_code_runinfo_sas')
    }
    
    # Create output dataframe
    output_data = {'instance': df_combined['instance']}
    
    # Add comparison columns
    for col_name, (human_col, pred_col) in comparison_cols.items():
        if human_col in df_combined.columns and pred_col in df_combined.columns:
            output_data[f'{col_name}_human'] = df_combined[human_col]
            output_data[f'{col_name}_pred'] = df_combined[pred_col]
            # Mark mismatches
            output_data[f'{col_name}_match'] = df_combined[human_col] == df_combined[pred_col]
    
    output_df = pd.DataFrame(output_data)
    
    # Save to Excel
    output_file = f"results/{output_filename}.xlsx"
    output_df.to_excel(output_file, index=False, engine='openpyxl')
    
    # Load workbook and apply formatting
    wb = load_workbook(output_file)
    ws = wb.active
    
    # Define blue font for mismatches
    blue_font = Font(color="0000FF")
    
    # Apply formatting
    for row_idx in range(2, len(output_df) + 2):  # Start from row 2 (skip header)
        for col_name in comparison_cols.keys():
            if f'{col_name}_match' in output_data:
                match_col = None
                human_col = None
                pred_col = None
                
                # Find column indices
                for col_idx, header in enumerate(output_df.columns, 1):
                    if header == f'{col_name}_match':
                        match_col = col_idx
                    elif header == f'{col_name}_human':
                        human_col = col_idx
                    elif header == f'{col_name}_pred':
                        pred_col = col_idx
                
                # Check if it's a mismatch and apply blue font
                if match_col and human_col and pred_col:
                    match_value = ws.cell(row=row_idx, column=match_col).value
                    if match_value == False:  # Mismatch
                        ws.cell(row=row_idx, column=human_col).font = blue_font
                        ws.cell(row=row_idx, column=pred_col).font = blue_font
    
    wb.save(output_file)
    print(f"Detailed comparison saved to {output_file}")
    return output_file

def calculate_agreement_rate_between_judges():
    df_sas = pd.read_excel(f"sas/sas_outputs/results_parsed_summary_sas.xlsx", engine="openpyxl")
    df_llms = pd.read_excel(f"llms/llms_outputs/results_parsed_summary_llms.xlsx", engine="openpyxl")
    df_combined = df_sas.merge(df_llms, on='instance')

    df_sas_reversed = pd.read_excel(f"sas/sas_outputs/results_parsed_reversed_summary_sas.xlsx", engine="openpyxl")
    df_llms_reversed = pd.read_excel(f"llms/llms_outputs/results_parsed_reversed_summary_llms.xlsx", engine="openpyxl")
    df_combined_reversed = df_sas_reversed.merge(df_llms_reversed, on='instance')

    # calculate agreement rate between original and reversed for every model
    models = {
        'gemini 2.5 flash': ('gemini_2_5_flash_crash_detection_code', 'gemini_2_5_flash_crash_detection_code_runinfo'),
        'qwen 2.5 32b instruct': ('Qwen_2_5_32B_Instruct_crash_detection_code', 'Qwen_2_5_32B_Instruct_crash_detection_code_runinfo'),
        'gpt-5': ('gpt_5_crash_detection_code', 'gpt_5_crash_detection_code_runinfo'),
        'pylint': ('pylint_crash_detection_code', 'pylint_crash_detection_code_runinfo'),
        'pyright': ('pyright_crash_detection_code', 'pyright_crash_detection_code_runinfo')
    }
    results = []
    for model_name, (col_code, col_runinfo) in models.items():
        # For -RT (code only)
        if col_code in df_combined.columns and col_code in df_combined_reversed.columns:
            valid_comparisons = df_combined.dropna(subset=[col_code]).merge(
                df_combined_reversed[['instance', col_code]], on='instance', suffixes=('_orig', '_reversed'))
            matches_code = (valid_comparisons[f'{col_code}_orig'] == valid_comparisons[f'{col_code}_reversed']).sum()
        
        # For +RT (code + runinfo)
        if col_runinfo in df_combined.columns and col_runinfo in df_combined_reversed.columns:
            valid_comparisons_runinfo = df_combined.dropna(subset=[col_runinfo]).merge(
                df_combined_reversed[['instance', col_runinfo]], on='instance', suffixes=('_orig', '_reversed'))
            matches_runinfo = (valid_comparisons_runinfo[f'{col_runinfo}_orig'] == valid_comparisons_runinfo[f'{col_runinfo}_reversed']).sum()
        
        results.append({
            'Model': model_name,
            'Agreement_Rate': (matches_code + matches_runinfo)/(len(valid_comparisons) + len(valid_comparisons_runinfo))
        })
    # overall agreement rate
    total_matches = sum((df_combined[col] == df_combined_reversed[col]).sum() for col in df_combined.columns)
    total_comparisons = sum(len(df_combined) for col in df_combined.columns)
    results.append({
        'Model': 'Overall',
        'Agreement_Rate': total_matches / total_comparisons
    })
    results_df = pd.DataFrame(results)
    print("Agreement rate between original and reversed inputs:\n", results_df)
    # save to file
    results_df.to_excel("results/agreement_rate_between_judges.xlsx", index=False, engine='openpyxl')

def _load_and_prepare_data(metric_type='crash_detection'):
    """Load and prepare data for analysis"""
    df_human = pd.read_excel("results/results_human_evaluated.xlsx", sheet_name="Human_evaluation", engine="openpyxl")
    df_label = pd.read_excel("results/benchmark_labels.xlsx", engine="openpyxl")
    df_label.rename(columns={'nb_name': 'instance'}, inplace=True)
    
    if metric_type == 'crash_detection':
        # For crash detection: only _reproduced cases (remove _fixed suffix)
        df_human = df_human[~df_human['instance'].str.endswith('_fixed')]
        # modify instance values by removing _reproduced suffix
        df_human['instance'] = df_human['instance'].str.replace('_reproduced', '', regex=False)
    else:
        # For accuracy: include all instances (both _fixed and _reproduced)
        # Remove both _fixed and _reproduced suffixes to match benchmark labels
        df_human['instance'] = df_human['instance'].str.replace('_fixed', '', regex=False).str.replace('_reproduced', '', regex=False)
    
    # Merge dataframes on instance name
    df_combined = df_human.merge(df_label, on='instance', how='inner')
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

def _get_model_definitions():
    """Get model column definitions"""
    return {
        'Gemini 2.5 Flash': {
            'with_runinfo': 'gemini_2_5_flash_crash_detection_code_runinfo',
            'without_runinfo': 'gemini_2_5_flash_crash_detection_code'
        },
        'Qwen 2.5 32B Instruct': {
            'with_runinfo': 'Qwen_2_5_32B_Instruct_crash_detection_code_runinfo',
            'without_runinfo': 'Qwen_2_5_32B_Instruct_crash_detection_code'
        },
        'GPT-5': {
            'with_runinfo': 'gpt_5_crash_detection_code_runinfo',
            'without_runinfo': 'gpt_5_crash_detection_code'
        },
        'PyLint': {
            'with_runinfo': 'pylint_crash_detection_code_runinfo',
            'without_runinfo': 'pylint_crash_detection_code'
        },
        'PyRight': {
            'with_runinfo': 'pyright_crash_detection_code_runinfo',
            'without_runinfo': 'pyright_crash_detection_code'
        }
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
    else:
        fig_size = (12, 5.2)
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
            line_ax.set_ylim(-0.11, 0.21)
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
    models = _get_model_definitions()
    
    # Filter out Static Analysis tools if requested
    sa_tools = ['PyLint', 'PyRight']
    models = {k: v for k, v in models.items() if k not in sa_tools}
    
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
