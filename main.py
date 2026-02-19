# Results summary
from llms import result_check

# Generate detection only results
result_check.check_all_predictions_auto()

# ----------------------------------------------------------------------------------------------

# Final results statistics
from results import result_statistics
from llms.config_llms import config

current_task = "result parsing llm diagnosis only"
config.current_task = current_task

result_statistics.calculate_cohens_kappa()
# ----------------------------------------------------------------------------------------------

# # overall plots
# from results.result_statistics import generate_label_results_statistics

# generate_label_results_statistics('Libs-cause', 'accuracy')
# generate_label_results_statistics('Libs-cause', 'crash_detection')
# generate_label_results_statistics('label_root_cause', 'crash_detection')


# # ----------------------------------------------------------------------------------------------
# # ---------------------------------early crash detection statistics-----------------------------
# # ----------------------------------------------------------------------------------------------
# # Count runtime of prior cells (executed code cells)
# from llms import prompt_extractor
# from llms.config_llms import config
# from llms import result_check
# import json
# from pathlib import Path

# # rerun the following cases only
# lib_names = ["sklearn"]
# # lib_names = ["tensorflow", "torch", "sklearn", "pandas", "numpy", "NBspecific", "other"]

# for lib_name in lib_names:
#     res = prompt_extractor.runtime_count_prior_cells(lib_name)

#     # save to file incrementally
#     output_path = Path("runtime_counts_prior_cells.json")
#     try:
#         with open(output_path, 'r') as f:
#             existing_data = json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError):
#         existing_data = {}
#     existing_data[lib_name] = res
#     with open(output_path, 'w') as f:
#         json.dump(existing_data, f, indent=4)

# # ----------------------------------------------------------------------------------------------
# # measure query time of LLM (GPT-5)
# from llms.config_llms import config
# import os, json
# from llms import llm_executor
# from llms.huggingface_model_loader import get_qwen_model
# from pathlib import Path

# # settings
# llm_server = "openai_gpt" 
# llm_model = "gpt-5"
# current_task = "crash detection with executed code cells and runinfo"

# lib_names = ["tensorflow", "torch", "numpy", "sklearn", "pandas", "NBspecific", "other"]

# config.current_task = current_task
# config.current_llm_model = llm_model
# print(f"Current task: {config.current_task}")
# print(f"Current LLM model: {config.current_llm_model}")

# for lib_name in lib_names:
#     res_lib = {}
#     for filename in os.listdir(config.path_input.joinpath(lib_name)):
#         exec_llm = llm_executor.LLMExecutor(model=llm_model, libname = lib_name, filename=filename)
#         res = exec_llm.record_time_llm_openai()
#         res_lib[filename] = res
#     # save to file incrementally
#     output_path = Path("runtime_counts_cranellm_gpt5.json")
#     try:
#         with open(output_path, 'r') as f:
#             existing_data = json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError):
#         existing_data = {}
#     existing_data[lib_name] = res_lib
#     with open(output_path, 'w') as f:
#         json.dump(existing_data, f, indent=4)

# ---------------------------------------------------------------------------------------------- 
# calculate statistics of the execution time
# ---------------------------------------------------------------------------------------------- 
# # calculate statistics of the execution time
# from pathlib import Path
# import json, os
# import statistics

# def calculate_statistics(data_list, name):
#     """Calculate and print statistics for a list of numbers"""
#     if not data_list:
#         print(f"{name}: No data available")
#         return {}
    
#     stats = {
#         'count': len(data_list),
#         'mean': statistics.mean(data_list),
#         'median': statistics.median(data_list),
#         'std': statistics.stdev(data_list) if len(data_list) > 1 else 0,
#         'min': min(data_list),
#         'max': max(data_list),
#         'sum': sum(data_list)
#     }
    
#     print(f"\n{name} Statistics:")
#     print(f"Count: {stats['count']}")
#     print(f"Mean: {stats['mean']:.4f}")
#     print(f"Median: {stats['median']:.4f}")
#     print(f"Std Dev: {stats['std']:.4f}")
#     print(f"Min: {stats['min']:.4f}")
#     print(f"Max: {stats['max']:.4f}")
#     print(f"Sum: {stats['sum']:.4f}")
    
#     return stats

# # Read both JSON files first
# with open(Path("runtime_counts_prior_cells.json"), 'r') as f:
#     time_execution = json.load(f)

# with open(Path("runtime_counts_cranellm_gpt5.json"), 'r') as f:
#     time_query = json.load(f)

# # Extract execution times from prior cells
# execution_times = []
# for lib_name, lib_data in time_execution.items():
#     if isinstance(lib_data, dict):
#         for filename, time_val in lib_data.items():
#             if isinstance(time_val, (int, float)):
#                 execution_times.append(time_val)
#     elif isinstance(lib_data, (int, float)):
#         execution_times.append(lib_data)

# # Extract query times (only keys with "reproduced" in them)
# query_times = []
# for lib_name, lib_data in time_query.items():
#     if isinstance(lib_data, dict):
#         for filename, time_val in lib_data.items():
#             if "reproduced" in filename.lower() and isinstance(time_val, (int, float)):
#                 query_times.append(time_val)

# # Calculate statistics for both datasets
# execution_stats = calculate_statistics(execution_times, "Execution Time (Prior Cells)")
# query_stats = calculate_statistics(query_times, "Query Time (GPT-5, Reproduced Only)")

# # Calculate saved time statistics
# saved_times = []

# # Match reproduced files between datasets
# for lib_name in time_execution.keys():
#     if lib_name in time_query:
#         exec_lib_data = time_execution[lib_name]
#         query_lib_data = time_query[lib_name]
        
#         if isinstance(exec_lib_data, dict) and isinstance(query_lib_data, dict):
#             for exec_filename in exec_lib_data.keys():
#                 if "reproduced" in exec_filename.lower():
#                     # Try to match with and without .txt extension
#                     query_filename = exec_filename + '.txt'
#                     if query_filename in query_lib_data:
#                         exec_time = exec_lib_data[exec_filename]
#                         query_time = query_lib_data[query_filename]
#                         if isinstance(exec_time, (int, float)) and isinstance(query_time, (int, float)):
#                             saved_time = exec_time - query_time
#                             saved_times.append(saved_time)

# print(f"\nFinal saved_times length: {len(saved_times)}")
# saved_stats = calculate_statistics(saved_times, "Saved Time (Execution - Query)")

# # Summary
# print(f"\n{'='*50}")
# print("SUMMARY:")
# print(f"{'='*50}")
# print(f"Total saved time calculations: {sum(saved_times)}")

# if saved_times:
#     positive_savings = [t for t in saved_times if t > 0]
#     negative_savings = [t for t in saved_times if t < 0]
#     print(f"Cases with time savings (positive): {len(positive_savings)}")
#     print(f"Cases with time loss (negative): {len(negative_savings)}")
    
#     if positive_savings:
#         print(f"Average time saved (positive cases): {statistics.mean(positive_savings):.4f}")
#     if negative_savings:
#         print(f"Average time loss (negative cases): {statistics.mean(negative_savings):.4f}")

# ---------------------------------------------------------------------------------------------- 
# calculate incremental input token count caused by adding API documentation
# ----------------------------------------------------------------------------------------------

# from utils.token_analysis import run_token_analysis

# run_token_analysis()

# ----------------------------------------------------------------------------------------------
# sample a representative set of cases for manual analysis
# ----------------------------------------------------------------------------------------------
# from utils import manual_analysis_sampler as mas

# mas.run_manual_analysis_sampling()
# mas.filter_files_by_sample()

