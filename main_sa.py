# # generate typing annotations from runinfo
# from llms.config_llms import config as llmconfig
# import sas.runtime_injector as runtime_injector
# from pathlib import Path

# sas_runinfo_path = Path("sas/sas_inputs/executed_code_runinfo")
# # lib_names = ["tensorflow", "torch", "sklearn", "pandas", "numpy", "NBspecific", "other"] #"tensorflow", "torch", "sklearn", "pandas", "numpy", "NBspecific", "other"
# # for lib_name in lib_names:
# #     runtime_injector.generate_runinfo_injected_version(sas_runinfo_path, lib_name)

# # rerun the following cases only
# lib_case_names = {
#     "torch": ["torch_1", "torch_5"],
#     "sklearn": ["sklearn_6", "sklearn_10"],
#     "pandas": ["pandas_14"],
#     "numpy": ["numpy_7", "numpy_8"],
#     "other": ["torchvision_1"],
# }
# for lib_name, case_names in lib_case_names.items():
#     runtime_injector.generate_runinfo_injected_version(sas_runinfo_path, lib_name, case_names=case_names)

# ----------------------------------------------------------------------------------------------

# # using static analyzers such as pylint and pyright
# from pathlib import Path
# from llms.config_llms import config as llmconfig
# from sas import static_analysis_tools as sat

# # current_task = "crash detection with executed code cells and runinfo" # and runinfo
# # lib_names = ["tensorflow", "torch", "sklearn", "pandas", "numpy", "NBspecific", "other"] # "tensorflow", "torch", "sklearn", "pandas", "numpy", "NBspecific", "other"

# # llmconfig.current_task = current_task
# # print(f"Current task: {llmconfig.current_task}")
# # for libname in lib_names:
# #     for saname in ["pyright", "pylint"]: #"pyright", "pylint"
# #         if "runinfo" in llmconfig.current_task:
# #             path_input = Path(f'sas/sas_inputs/executed_code_runinfo/{libname}')
# #             path_output = Path(f'sas/sas_outputs/{saname}/crash_detection_code_runinfo/{libname}')
# #         else:
# #             path_input = Path(f'llms/llms_inputs/executed_code/{libname}')
# #             path_output = Path(f'sas/sas_outputs/{saname}/crash_detection_code/{libname}')
# #         print(f"Runing {saname} on {libname}")
# #         sat.run_sas_all(saname, path_input, path_output)

# # rerun the following cases only
# lib_case_names = {
#     "sklearn": ["sklearn_10"],
#     # "numpy": ["numpy_8"],
# }
# tasks = ["crash detection with executed code cells", "crash detection with executed code cells and runinfo"]
# for current_task in tasks:
#     llmconfig.current_task = current_task
#     print(f"Current task: {llmconfig.current_task}")
#     for lib_name, case_names in lib_case_names.items():
#         for saname in ["pyright", "pylint"]: #"pyright", "pylint"
#             if "runinfo" in llmconfig.current_task:
#                 path_input = Path(f'sas/sas_inputs/executed_code_runinfo/{lib_name}')
#                 path_output = Path(f'sas/sas_outputs/{saname}/crash_detection_code_runinfo/{lib_name}')
#             else:
#                 path_input = Path(f'llms/llms_inputs/executed_code/{lib_name}')
#                 path_output = Path(f'sas/sas_outputs/{saname}/crash_detection_code/{lib_name}')
#             print(f"Runing {saname} on {lib_name} for selected cases only")
#             sat.run_sas_all(saname, path_input, path_output, case_names=case_names)
# ----------------------------------------------------------------------------------------------

# # result parsing via LLM as a judge
# # Using local server - Selene-1-Mini-Llama-3.1-8B from Huggingface

# from llms.huggingface_model_loader import get_qwen_model
# from llms import result_check
# from llms.config_llms import config
# from pathlib import Path

# lib_names = ["tensorflow", "torch", "sklearn", "pandas", "numpy", "NBspecific", "other"] #"tensorflow", "torch", "sklearn", "pandas", "numpy", "NBspecific", "other"
# target_checking_sas = ["pyright", "pylint"]

# current_task = "result parsing sa"
# config.current_task = current_task
# print(f"Current task: {config.current_task}")
# # if_reverse = False #True

# llm_model = "AtlaAI/Selene-1-Mini-Llama-3.1-8B"
# tokenizer, model = get_qwen_model(llm_model)

# for if_reverse in [False, True]:
#     for target_sa in target_checking_sas:
#         if if_reverse:
#             config.path_res_parsed_res_sa = Path("sas/sas_outputs/results_parsed_reversed/")
#         else:
#             config.path_res_parsed_res_sa = Path("sas/sas_outputs/results_parsed/")
#         result_check.check_all_predictions_for_sa(target_sa, tokenizer, model, lib_names=lib_names, if_reverse=if_reverse)

# ----------------------------------------------------------------------------------------------

# # Results summary
# from llms import result_check
# from llms.config_llms import config
# from pathlib import Path

# current_task = "result parsing sa"
# config.current_task = current_task
# print(f"Current task: {config.current_task}")
# # if_reverse = True
# for if_reverse in [False, True]:
#     if not if_reverse:
#         config.path_res_parsed_res_sa = Path("sas/sas_outputs/results_parsed/")
#         output_path = "sas/sas_outputs/results_parsed_summary_sas.xlsx"
#     else:
#         config.path_res_parsed_res_sa = Path("sas/sas_outputs/results_parsed_reversed/")
#         output_path = "sas/sas_outputs/results_parsed_reversed_summary_sas.xlsx"

#     target_checking_sas = ["pyright", "pylint"] 
#     lib_names = ["tensorflow", "torch", "sklearn", "pandas", "numpy", "NBspecific", "other"]

#     result_check.check_all_parsed_results_sa(target_checking_sas, output_path, lib_names=lib_names)
