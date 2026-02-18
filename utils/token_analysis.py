"""
Token Analysis Module for API Documentation Impact

This module analyzes the token count differences between baseline prompts
and prompts enhanced with API documentation.
"""

import tiktoken
import os
import statistics
import numpy as np
from pathlib import Path


def count_tokens(text, tokenizer):
    """Count tokens in text using the provided tokenizer."""
    return len(tokenizer.encode(text, disallowed_special=()))


def analyze_token_differences(baseline_path, docs_path, output_path, tokenizer_name="o200k_base"):
    """
    Analyze token count differences between baseline and documentation-enhanced files.
    
    Args:
        baseline_path (Path): Path to baseline files directory
        docs_path (Path): Path to documentation-enhanced files directory
        output_path (Path): Path for output analysis file
        tokenizer_name (str): Tokenizer encoding name (default: o200k_base for GPT-4o/GPT-5)
    
    Returns:
        dict: Dictionary containing analysis statistics
    """
    # Load tokenizer
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    
    # Store all delta token counts and baseline data
    all_delta_tokens = []
    all_baseline_tokens = []
    all_percentage_changes = []
    
    # Get all txt files from both directories
    baseline_files = {f.relative_to(baseline_path): f for f in baseline_path.rglob("*.txt")}
    doc_files = {f.relative_to(docs_path): f for f in docs_path.rglob("*.txt")}
    
    # Find common files
    common_files = set(baseline_files.keys()) & set(doc_files.keys())
    
    # Prepare output content
    output_lines = []
    output_lines.append(f"Found {len(baseline_files)} baseline files, {len(doc_files)} doc files, {len(common_files)} common files\n")
    
    for relative_path in common_files:
        # Read baseline file
        with open(baseline_files[relative_path], 'r', encoding='utf-8') as f:
            input_baseline = f.read()
        
        # Read file with docs
        with open(doc_files[relative_path], 'r', encoding='utf-8') as f:
            input_with_docs = f.read()
        
        # Calculate tokens
        tokens_baseline = count_tokens(input_baseline, tokenizer)
        tokens_with_docs = count_tokens(input_with_docs, tokenizer)
        
        delta_tokens = tokens_with_docs - tokens_baseline
        percentage_change = (delta_tokens / tokens_baseline * 100) if tokens_baseline > 0 else 0
        
        all_delta_tokens.append(delta_tokens)
        all_baseline_tokens.append(tokens_baseline)
        all_percentage_changes.append(percentage_change)
        
        output_lines.append(f"{relative_path}: baseline={tokens_baseline}, with_docs={tokens_with_docs}, delta={delta_tokens} ({percentage_change:+.1f}%)\n")
    
    # Calculate and report statistics
    stats_dict = {}
    if all_delta_tokens:
        output_lines.append(f"\n{'='*60}\n")
        output_lines.append("TOKEN INCREASE STATISTICS (Adding API Documentation)\n")
        output_lines.append(f"{'='*60}\n")
        output_lines.append(f"Total files processed: {len(all_delta_tokens)}\n")
        output_lines.append(f"Mean delta tokens: {statistics.mean(all_delta_tokens):.2f}\n")
        output_lines.append(f"Median delta tokens: {statistics.median(all_delta_tokens):.2f}\n")
        output_lines.append(f"75th percentile: {np.percentile(all_delta_tokens, 75):.2f}\n")
        output_lines.append(f"90th percentile: {np.percentile(all_delta_tokens, 90):.2f}\n")
        output_lines.append(f"95th percentile: {np.percentile(all_delta_tokens, 95):.2f}\n")
        output_lines.append(f"Min delta tokens: {min(all_delta_tokens)}\n")
        output_lines.append(f"Max delta tokens: {max(all_delta_tokens)}\n")
        output_lines.append(f"Total additional tokens across all files: {sum(all_delta_tokens)}\n")
        
        # Baseline token statistics
        output_lines.append(f"\nBaseline Token Statistics:\n")
        output_lines.append(f"Mean baseline tokens: {statistics.mean(all_baseline_tokens):.2f}\n")
        output_lines.append(f"Median baseline tokens: {statistics.median(all_baseline_tokens):.2f}\n")
        output_lines.append(f"Total baseline tokens across all files: {sum(all_baseline_tokens)}\n")
        
        # Percentage change statistics
        output_lines.append(f"\nPercentage Change Statistics:\n")
        output_lines.append(f"Mean percentage change: {statistics.mean(all_percentage_changes):.2f}%\n")
        output_lines.append(f"Median percentage change: {statistics.median(all_percentage_changes):.2f}%\n")
        output_lines.append(f"75th percentile: {np.percentile(all_percentage_changes, 75):.2f}%\n")
        output_lines.append(f"90th percentile: {np.percentile(all_percentage_changes, 90):.2f}%\n")
        output_lines.append(f"95th percentile: {np.percentile(all_percentage_changes, 95):.2f}%\n")
        output_lines.append(f"Min percentage change: {min(all_percentage_changes):.2f}%\n")
        output_lines.append(f"Max percentage change: {max(all_percentage_changes):.2f}%\n")
        
        # Overall percentage increase
        total_baseline = sum(all_baseline_tokens)
        total_with_docs = total_baseline + sum(all_delta_tokens)
        overall_percentage = (sum(all_delta_tokens) / total_baseline * 100) if total_baseline > 0 else 0
        output_lines.append(f"Overall percentage increase: {overall_percentage:.2f}%\n")
        
        # Additional insights
        positive_deltas = [d for d in all_delta_tokens if d > 0]
        negative_deltas = [d for d in all_delta_tokens if d < 0]
        zero_deltas = [d for d in all_delta_tokens if d == 0]
        
        output_lines.append(f"\nDistribution:\n")
        output_lines.append(f"Files with increased tokens: {len(positive_deltas)} ({len(positive_deltas)/len(all_delta_tokens)*100:.1f}%)\n")
        output_lines.append(f"Files with decreased tokens: {len(negative_deltas)} ({len(negative_deltas)/len(all_delta_tokens)*100:.1f}%)\n")
        output_lines.append(f"Files with no change: {len(zero_deltas)} ({len(zero_deltas)/len(all_delta_tokens)*100:.1f}%)\n")
        
        if positive_deltas:
            positive_percentages = [all_percentage_changes[i] for i, d in enumerate(all_delta_tokens) if d > 0]
            output_lines.append(f"Average increase (positive cases): {statistics.mean(positive_deltas):.2f} tokens ({statistics.mean(positive_percentages):.2f}%)\n")
        if negative_deltas:
            negative_percentages = [all_percentage_changes[i] for i, d in enumerate(all_delta_tokens) if d < 0]
            output_lines.append(f"Average decrease (negative cases): {statistics.mean(negative_deltas):.2f} tokens ({statistics.mean(negative_percentages):.2f}%)\n")
        
        # Store statistics in dictionary for return
        stats_dict = {
            'total_files': len(all_delta_tokens),
            'mean_delta': statistics.mean(all_delta_tokens),
            'median_delta': statistics.median(all_delta_tokens),
            'total_additional_tokens': sum(all_delta_tokens),
            'overall_percentage_increase': overall_percentage,
            'baseline_stats': {
                'mean': statistics.mean(all_baseline_tokens),
                'median': statistics.median(all_baseline_tokens),
                'total': sum(all_baseline_tokens)
            },
            'distribution': {
                'increased': len(positive_deltas),
                'decreased': len(negative_deltas),
                'unchanged': len(zero_deltas)
            }
        }
    else:
        output_lines.append("No files were processed successfully!\n")
        stats_dict = {'error': 'No files processed'}
    
    # Write results to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    return stats_dict


def run_token_analysis():
    """Run the complete token analysis with default paths."""
    # Define paths
    baseline_path = Path("llms/llms_inputs/executed_code_runinfo_full/")
    docs_path = Path("llms/llms_inputs/executed_code_runinfo_full_doc/")
    output_path = Path("results/runtime_doc_token_analysis.txt")
    
    # Run analysis
    stats = analyze_token_differences(baseline_path, docs_path, output_path)
    
    print(f"Token analysis complete. Results written to {output_path}")
    return stats


if __name__ == "__main__":
    # Run analysis if script is executed directly
    run_token_analysis()