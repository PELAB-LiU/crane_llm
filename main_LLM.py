# Prompt generation
# Executed code + target cell, or executed code with runinfo + target cell, depending on the task

from llms import prompt_extractor
from llms.config_llms import config
from llms import result_check

# current_task = "crash detection with executed code cells"
current_task = "crash detection with executed code cells and runinfo"
config.current_task = current_task
print(f"Current task: {config.current_task}")

lib_names = ["tensorflow", "torch", "sklearn", "numpy", "pandas", "NBspecific", "other"] #
for lib_name in lib_names:
    prompt_extractor.generate_prompt(lib_name, generate_all_combinations=True) #, force_regenerate=True)

# # rerun the following cases only
# lib_case_names = {
#     "sklearn": ["sklearn_3"],
# }
# for lib_name, case_names in lib_case_names.items():
#     prompt_extractor.generate_prompt(lib_name, case_names=case_names)

# ----------------------------------------------------------------------------------------------

# Predict if a target cell in a Jupyter notebook will crash or not, with bug allocation.
# Using OpenAI API / Google Gemini API / local server - Qwen model from Huggingface
from llms.config_llms import config
import os, json
from llms import llm_executor
from llms.huggingface_model_loader import get_qwen_model

# settings
runs = 5
llm_server = "google_gemini" # "openai_gpt", "google_gemini", "local_huggingface"
llm_model = "gemini-2.5-flash" # "gpt-5", "gemini-2.5-flash", "Qwen/Qwen2.5-Coder-32B-Instruct"
if llm_server == "local_huggingface": 
    tokenizer, model = get_qwen_model(llm_model)
tasks_to_run = [
    # "crash detection with executed code cells",
    "crash detection with executed code cells and runinfo"
]
ablations_to_run = ["r_v", "s_r", "s_v"] # "full", "r_v", "s_r", "s_v"
lib_names = ["tensorflow", "torch", "numpy", "sklearn", "pandas", "NBspecific", "other"] # 
for current_task in tasks_to_run:
    config.current_task = current_task
    config.current_llm_model = llm_model
    print(f"Current task: {config.current_task}")
    print(f"Current LLM model: {config.current_llm_model}")
    # print(config.prompt_instruct)
    for ablation_setting in ablations_to_run:
        config.current_ablation_setting = ablation_setting
        print(f"Current ablation setting: {config.current_ablation_setting}")
        # optional setting for using API documentations
        # config.current_doc = True if "runinfo" in current_task and "full" in ablation_setting else False
        for lib_name in lib_names:
            id_crash = 0
            for i in range(1, runs+1, 1):
                for filename in os.listdir(config.path_input.joinpath(lib_name)):
                    # # covered previously failed rounds:
                    # check_outputfile = config.path_res.joinpath(f"{filename.split('.')[0]}.json")
                    # json_output = json.load(open(check_outputfile, 'r'))
                    # if len(json_output)>=i:
                    #     continue
                    exec_llm = llm_executor.LLMExecutor(model=llm_model, libname = lib_name, filename=filename)
                    if llm_server == "openai_gpt":
                        exec_llm.llm_multiple_rounds_openai()
                    elif llm_server == "google_gemini":
                        exec_llm.llm_multiple_rounds_gemini()
                    elif llm_server == "local_huggingface": 
                        exec_llm.llm_multiple_rounds_huggingface(tokenizer, model)
                    id_crash += 1
                print(f"Number {i} round: Successfully detected {id_crash} cases")


# # rerun the following cases only
# llm_server = "local_huggingface"
# llm_model = "Qwen/Qwen2.5-Coder-32B-Instruct"
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

