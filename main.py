# # Generate detection only results
# from llms import result_check

# result_check.check_all_predictions_auto()

# ----------------------------------------------------------------------------------------------

# # Final results statistics
# # LLM as a judge evaluation
# from results import result_statistics

# for if_reversed in [True, False]:
#     if if_reversed:
#         filename_sas = "results_parsed_reversed_summary_sas"
#         filename_llms = "results_parsed_reversed_summary_llms"
#         filename_output = "llm_judge_statistics_reversed"
#         filename_output_detailed = "llm_judge_detailed_comparisons_reversed"
#     else:
#         filename_sas = "results_parsed_summary_sas"
#         filename_llms = "results_parsed_summary_llms"
#         filename_output = "llm_judge_statistics"
#         filename_output_detailed = "llm_judge_detailed_comparisons"
#     result_statistics.generate_llm_judge_result_statistics(filename_sas, filename_llms, filename_output)
#     result_statistics.create_detailed_comparison_excel(filename_sas, filename_llms, filename_output_detailed)

# ----------------------------------------------------------------------------------------------

# # Agreement rate between LLM judge with its reversed inputs
# from results import result_statistics
# result_statistics.calculate_agreement_rate_between_judges()

# ----------------------------------------------------------------------------------------------

# # overall plots
# from results.result_statistics import generate_label_results_statistics

# generate_label_results_statistics('Libs-cause', 'accuracy')
# generate_label_results_statistics('Libs-cause', 'crash_detection')
# generate_label_results_statistics('label_root_cause', 'crash_detection')
