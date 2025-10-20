# # Prompt generation
# # Executed code + target cell, or executed code with runinfo + target cell, depending on the task

# from llms import prompt_extractor
# from llms.config_llms import config
# from llms import result_check

# # current_task = "crash detection with executed code cells"
# current_task = "crash detection with executed code cells and runinfo"
# config.current_task = current_task
# print(f"Current task: {config.current_task}")

# # lib_names = ["other"]
# # for lib_name in lib_names:
# #     prompt_extractor.generate_prompt(lib_name)

# # rerun the following cases only
# lib_case_names = {
#     "NBspecific": ["NBspecific_7"],
# }
# for lib_name, case_names in lib_case_names.items():
#     prompt_extractor.generate_prompt(lib_name, case_names=case_names, force_regenerate=True)

# ----------------------------------------------------------------------------------------------

# # Predict if a target cell in a Jupyter notebook will crash or not, with bug allocation.
# # Using OpenAI API / Google Gemini API / local server - Qwen model from Huggingface
# from llms.config_llms import config
# import os, json
# from llms import llm_executor
# from llms.huggingface_model_loader import get_qwen_model

# # settings
# runs = 5
# # llm_server = "google_gemini" # "openai_gpt", "google_gemini", "local_huggingface"
# # llm_model = "gemini-2.5-flash" # "gpt-5", "gemini-2.5-flash", "Qwen/Qwen2.5-32B-Instruct"
# # if llm_server == "local_huggingface": 
# #     tokenizer, model = get_qwen_model(llm_model)
# # tasks_to_run = [
# #     "crash detection with executed code cells",
# #     "crash detection with executed code cells and runinfo"
# # ]
# # lib_names = ["tensorflow", "torch", "numpy", "sklearn", "pandas", "NBspecific", "other"] # 
# # for current_task in tasks_to_run:
# #     config.current_task = current_task
# #     config.current_llm_model = llm_model
# #     print(f"Current task: {config.current_task}")
# #     print(f"Current LLM model: {config.current_llm_model}")
# #     # print(config.prompt_instruct)

# #     for lib_name in lib_names:
# #         id_crash = 0
# #         for i in range(1, runs+1, 1):
# #             for filename in os.listdir(config.path_input.joinpath(lib_name)):
# #                 # # covered previously failed rounds:
# #                 # check_outputfile = config.path_res.joinpath(f"{filename.split('.')[0]}.json")
# #                 # json_output = json.load(open(check_outputfile, 'r'))
# #                 # if len(json_output)>=i:
# #                 #     continue
# #                 exec_llm = llm_executor.LLMExecutor(model=llm_model, libname = lib_name, filename=filename)
# #                 if llm_server == "openai_gpt":
# #                     exec_llm.llm_multiple_rounds_openai()
# #                 elif llm_server == "google_gemini":
# #                     exec_llm.llm_multiple_rounds_gemini()
# #                 elif llm_server == "local_huggingface": 
# #                     exec_llm.llm_multiple_rounds_huggingface(tokenizer, model)
# #                 id_crash += 1
# #             print(f"Number {i} round: Successfully detected {id_crash} cases")


# # rerun the following cases only
# llm_server = "local_huggingface"
# llm_model = "Qwen/Qwen2.5-32B-Instruct"
# if llm_server == "local_huggingface": 
#     tokenizer, model = get_qwen_model(llm_model)
# current_task = "crash detection with executed code cells"
# lib_case_names = {
#     "pandas": ["pandas_2_reproduced", "pandas_13_reproduced"],
#     "numpy": ["numpy_7_fixed", "numpy_4_reproduced"], # numpy_4_reproduced should
#     "tensorflow": ["tensorflow_14_reproduced"],
#     "torch": ["torch_5_fixed", "torch_10_reproduced"],
#     "sklearn": ["sklearn_3_fixed", "sklearn_10_fixed"],
#     "NBspecific": ["NBspecific_3_reproduced", "NBspecific_5_reproduced"], # NBspecific_3_reproduced, NBspecific_5_reproduced should
#     "other": ["seaborn_3_fixed", "seaborn_3_reproduced"],
# }
# config.current_task = current_task
# config.current_llm_model = llm_model
# print(f"Current task: {config.current_task}")
# print(f"Current LLM model: {config.current_llm_model}")

# for i in range(1, runs+1, 1):
#     id_crash = 0
#     for lib_name, case_names in lib_case_names.items():
#         for filename in os.listdir(config.path_input.joinpath(lib_name)):
#             if filename.split(".")[0] not in case_names:
#                continue
#             exec_llm = llm_executor.LLMExecutor(model=llm_model, libname = lib_name, filename=filename)
#             if llm_server == "openai_gpt":
#                 exec_llm.llm_multiple_rounds_openai()
#             elif llm_server == "google_gemini":
#                 exec_llm.llm_multiple_rounds_gemini()
#             elif llm_server == "local_huggingface": 
#                 exec_llm.llm_multiple_rounds_huggingface(tokenizer, model)
#             id_crash += 1
#     print(f"Number {i} round: Successfully detected {id_crash} cases")

# ----------------------------------------------------------------------------------------------

# # Results
# # Results parsing - LLM as a Judge - 5 runs
# # Using local server - Selene-1-Mini-Llama-3.1-8B from Huggingface
# from llms.huggingface_model_loader import get_qwen_model
# from llms import result_check
# from llms.config_llms import config
# from pathlib import Path

# lib_names = ["tensorflow", "torch", "sklearn", "pandas", "numpy", "NBspecific", "other"]
# target_checking_llms = ["gemini_2_5_flash", "Qwen_2_5_32B_Instruct", "gpt_5"] # , "Qwen_2_5_32B_Instruct", "gpt_5"

# current_task = "result parsing llm"
# config.current_task = current_task
# print(f"Current task: {config.current_task}")
# if_reverse = True # False # True

# llm_model = "AtlaAI/Selene-1-Mini-Llama-3.1-8B"
# tokenizer, model = get_qwen_model(llm_model)

# for if_reverse in [True, False]:
#     for i in range(5):
#         print(f"Round {i+1} of checking results for LLMs...")
#         if if_reverse:
#             config.path_res_parsed_res_llm = Path(f"llms/llms_outputs/results_parsed_reversed_{i+1}")
#         else:
#             config.path_res_parsed_res_llm = Path(f"llms/llms_outputs/results_parsed_{i+1}")
#         for target_llm in target_checking_llms:
#             result_check.check_all_predictions(target_llm, tokenizer, model, lib_names=lib_names, if_reverse=if_reverse)

# ----------------------------------------------------------------------------------------------

# # Results summary for 5 runs
# from llms import result_check
# from llms.config_llms import config
# from pathlib import Path

# current_task = "result parsing llm"
# config.current_task = current_task
# print(f"Current task: {config.current_task}")
# # if_reverse = True

# target_checking_llms = ["gemini_2_5_flash", "Qwen_2_5_32B_Instruct", "gpt_5"]
# lib_names = ["tensorflow", "torch", "sklearn", "pandas", "numpy", "NBspecific", "other"]

# for if_reverse in [True, False]:
#     for i in range(5):
#         print(f"Round {i+1} of aggregating results for LLMs...")
#         if if_reverse:
#             config.path_res_parsed_res_llm = Path(f"llms/llms_outputs/results_parsed_reversed_{i+1}")
#             output_path = f"llms/llms_outputs/results_parsed_reversed_summary_llms_{i+1}.xlsx"
#         else:
#             config.path_res_parsed_res_llm = Path(f"llms/llms_outputs/results_parsed_{i+1}")
#             output_path = f"llms/llms_outputs/results_parsed_summary_llms_{i+1}.xlsx"
        
#         result_check.check_all_parsed_results(target_checking_llms, output_path, lib_names=lib_names)

# ----------------------------------------------------------------------------------------------

# # Results aggregation from the 5 runs
# from llms import result_check

# # if_reverse = True
# for if_reverse in [True, False]:
#     if if_reverse:
#         filename = "results_parsed_reversed_summary_llms"
#     else:
#         filename = "results_parsed_summary_llms"

#     result_check.aggregate_parsed_result_summaries(filename)