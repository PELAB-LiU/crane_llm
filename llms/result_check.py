import json
import os
import pprint
import pandas as pd
import re
from llms.config_llms import config
from llms import llm_executor
from collections import Counter
from pathlib import Path

# LLM result parsing
def load_ground_truth():
    df = pd.read_excel(config.path_ground_truth, sheet_name="Ground_truth")
    return df

def find_ground_truth(df, instance_name):
    gt = df[df["instance"]==instance_name]
    res = ""
    if len(gt)==1:
        detection = gt.iloc[0]["detection"]
        reason = gt.iloc[0]["diagnosis"]
        if pd.notna(reason) and reason: # append reasoning
            if config.current_task == "result parsing llm diagnosis only":
                res = f"\nGround truth reasons: {reason}. "
            else:
                res = f"\nGround truth: {detection}. "
                res += reason
        elif detection is True: # buggy instance but no reasoning
            print(f"Buggy instance lacks ground truth of reasoning: {instance_name}")
    else:
        print(f"Instance not found in ground truth file: {instance_name}")
    return res

def _normalize_res(res):
    # First try to extract from brackets
    normalized = re.search(r'\[(.*?)\]', str(res))
    if normalized:
        normalized = normalized.group(1).strip().lower()
    else:
        # If no brackets found, try to match the response directly
        res_lower = str(res).strip().lower()
        valid_labels = ["correct", "partially correct", "reasoning wrong", "wrong"]
        
        # Check if the response matches any valid label
        for label in valid_labels:
            if label in res_lower:
                normalized = label
                break
        else:
            return f"Cannot extract detection labels: {res}"
        
    mapping = {
        "correct": "[Correct]",
        "partially correct": "[Partially correct]",
        "reasoning wrong": "[Reasoning Wrong]",
        "wrong": "[Wrong]"
    }

    if normalized in mapping:
        res = mapping[normalized]
        return res
    else:
        return f"Cannot verify: {res}"

def check_prediction(instance_name, df, query_llm_tokenizer, query_llm_model, llm_judge_name, input_json_file, output_json_file, if_reverse=False):
    predictions = load_json(input_json_file)
    parsed_ress = []
    # print(f"Predicting {input_json_file}...")
    for i, pred_res in enumerate(predictions):
        if config.current_task == "result parsing llm diagnosis only":
            if isinstance(pred_res, dict) and ("detection" in pred_res) and ("reasoning" in pred_res):
                if (pred_res["detection"] is True):
                    pred_res = pred_res["reasoning"]
                else:
                    print(f"Warning: LLM predicted non-buggy for {instance_name} at round {i}, skipping reasoning check: {pred_res}")
                    continue
            else:
                print(f"Warning: Unexpected LLM crash prediction format for {instance_name}: {pred_res}")
        if not if_reverse:
            user_message = find_ground_truth(df, instance_name) + f"\nPrediction from LLM: {pred_res}\n"
        else:
            user_message = f"\nPrediction from LLM: {pred_res}\n" + find_ground_truth(df, instance_name)
        # print(user_message)
        if (query_llm_tokenizer is not None) and (query_llm_model is not None):
            exec_llm = llm_executor.LLMExecutor(user_message = user_message)
            res = exec_llm.llm_run_huggingface(query_llm_tokenizer, query_llm_model)
        else:
            exec_llm = llm_executor.LLMExecutor(model=llm_judge_name, user_message = user_message)
            res = exec_llm.llm_run_openai()
        res = _normalize_res(res)
        if res.startswith("Cannot verify"):
            print(f"Warning: Cannot parse for instance {instance_name}: {res}")
        parsed_ress.append(res)

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
    # Save JSON
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(parsed_ress, f, indent=2, ensure_ascii=False)
    print(f"Saved parsed results to {output_json_file}")

def check_all_predictions(model_name, query_llm_tokenizer, query_llm_model, llm_judge_name, lib_names=None, if_reverse=False):
    df = load_ground_truth()
    llm_judge_abbr = config.get_model_abbr(llm_judge_name)
    if if_reverse:
        output_json_path = Path(str(config.path_res)+"_reversed"+llm_judge_abbr)
    else:
        output_json_path = Path(str(config.path_res)+f"{llm_judge_abbr}")
    for detection_mode in ["code", "runinfo"]: # "code", "runinfo"
        folder_name_llm = f"crash_detection_{model_name}_{detection_mode}"
        if "runinfo" in detection_mode:
            # get all folders start with folder_path_llm
            detection_mode_folders = [folder for folder in os.listdir(config.path_input) if folder.startswith(folder_name_llm)]
        else:
            detection_mode_folders = [folder_name_llm]
        for folder_name in detection_mode_folders:
            folder_path_llm = config.path_input.joinpath(folder_name)
            for filename in os.listdir(folder_path_llm):
                # if exists in output folder, skip
                output_json_file = output_json_path.joinpath(folder_name).joinpath(filename)
                if os.path.exists(output_json_file):
                    print(f"Output file {output_json_file} already exists, skipping...")
                    continue
                if (filename.endswith('.json')) and ((lib_names is None) or (filename.split('_')[0] in lib_names)):
                    instance_name = filename[:-len(".json")]
                    input_json_file = folder_path_llm.joinpath(filename)
                    check_prediction(instance_name, df, query_llm_tokenizer, query_llm_model, llm_judge_name, input_json_file, output_json_file, if_reverse=if_reverse)
                else:
                    print(f"Skipping file {filename} in {folder_path_llm} as it is not .json format.")
                # elif os.path.isdir(os.path.join(folder_path_llm, filename)) and (((lib_names is None) and (filename=="other")) or (filename in lib_names)):
                #     for sub_filename in os.listdir(os.path.join(folder_path_llm, filename)):
                #         if sub_filename.endswith('.json'):
                #             instance_name = sub_filename[:-len(".json")]
                #             input_json_file = folder_path_llm.joinpath(filename).joinpath(sub_filename)
                #             output_json_file = output_json_path.joinpath(folder_name).joinpath(filename).joinpath(sub_filename)
                #             check_prediction(instance_name, df, query_llm_tokenizer, query_llm_model, input_json_file, output_json_file, if_reverse=if_reverse)


def check_all_predictions_diagnosis_only(model_name, query_llm_tokenizer, query_llm_model, llm_judge_name, if_variants=False, if_reverse=False):
    df = load_ground_truth()
    llm_judge_abbr = config.get_model_abbr(llm_judge_name)
    if if_reverse:
        output_json_path = Path(str(config.path_res)+f"_diagnosis_reversed{llm_judge_abbr}")
    else:
        output_json_path =  Path(str(config.path_res)+f"_diagnosis{llm_judge_abbr}")
    df_diagnosis_only = check_all_predictions_auto(check_reasoning_flag=True, if_save_excel=False, if_variants=if_variants)
    for detection_mode in ["code", "runinfo"]: # "code", "runinfo"
        folder_name_llm = f"crash_detection_{model_name}_{detection_mode}"
        if if_variants and ("runinfo" in detection_mode):
            # get all folders start with folder_path_llm
            detection_mode_folders = [folder for folder in os.listdir(config.path_input) if folder.startswith(folder_name_llm)]
        else:
            detection_mode_folders = [folder_name_llm]
        n = 0
        for folder_name in detection_mode_folders:
            folder_path_llm = config.path_input.joinpath(folder_name)
            for filename in os.listdir(folder_path_llm):
                if (filename.endswith('.json')):
                    instance_name = filename[:-len(".json")]
                    # if should check reasoning for this instance
                    if folder_name in df_diagnosis_only.columns:
                        check_reasoning_mask = df_diagnosis_only[folder_name] == "check reasoning"
                        model_check_reasoning_instances = (df_diagnosis_only[check_reasoning_mask]['instance'].tolist())
                        if instance_name not in model_check_reasoning_instances:
                            print(f"Skipping instance {instance_name} as it does not require reasoning check.")
                            continue
                    # if exists in output folder, skip
                    output_json_file = output_json_path.joinpath(folder_name).joinpath(filename)
                    if os.path.exists(output_json_file):
                        print(f"Output file {output_json_file} already exists, skipping...")
                        continue
                    print(f"Processing instance {instance_name} for diagnosis check by llm judge {llm_judge_abbr}...")
                    n += 1
                    input_json_file = folder_path_llm.joinpath(filename)
                    check_prediction(instance_name, df, query_llm_tokenizer, query_llm_model, llm_judge_name, input_json_file, output_json_file, if_reverse=if_reverse)
                else:
                    print(f"Skipping file {filename} in {folder_path_llm} as it is not .json format.")
            print(f"{folder_name}: Processed {n} instances in for diagnosis check by llm judge {llm_judge_abbr}.")
                
def check_prediction_sa(instance_name, df, query_llm_tokenizer, query_llm_model, input_json_file, output_json_file, if_reverse=False):
    pred_res = load_json(input_json_file)
    # ground_truth = df[df["instance"]==instance_name].iloc[0]['detection']
    res = []
    # if (len(pred_res) <= 0) and (not ground_truth):
    #     res.append("[Correct]")
    # elif (len(pred_res) <= 0) and ground_truth:
    #     res.append("[Wrong]")
    # elif (len(pred_res) > 0) and (not ground_truth):
    #     res.append("[Wrong]")
    # else: # prediction is buggy and ground truth is buggy (true) -> compare the reasons with LLMs

    if not if_reverse:
        user_message = find_ground_truth(df, instance_name) + f"\nPrediction from static analyzer: {pred_res}\n"
    else:
        user_message = f"\nPrediction from static analyzer: {pred_res}\n" + find_ground_truth(df, instance_name)
    
    # print(user_message)
    print(f"Querying LLM as a judge for {instance_name} for 5 runs...")
    exec_llm = llm_executor.LLMExecutor(user_message = user_message)
    for i in range(5):
        res_op = exec_llm.llm_run_huggingface(query_llm_tokenizer, query_llm_model)
        res_nor = _normalize_res(res_op)
        if res_nor.startswith("Cannot verify"):
            print(f"Warning {i}th run: Cannot parse for instance {instance_name}: {res_nor}")
        res.append(res_nor)
        
    # print(f"Prediction from {input_json_file.name} is {res}.\nOriginal prediction:\n{pred_res}")
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
    # Save JSON
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"Saved parsed results to {output_json_file}")

def check_all_predictions_for_sa(sa_name, query_llm_tokenizer, query_llm_model, lib_names=None, if_reverse=False):
    df = load_ground_truth()
    input_folder_path = config.path_input.joinpath(sa_name)
    output_folder_path = config.path_res.joinpath(sa_name)
    for detection_mode in ["crash_detection_code", "crash_detection_code_runinfo"]:
        for lib_name in lib_names:
            folder_path_sa = input_folder_path.joinpath(detection_mode).joinpath(lib_name)
            for filename in os.listdir(folder_path_sa):
                if filename.endswith('.json'):
                    instance_name = filename[:-len(".json")]
                    input_json_file = folder_path_sa.joinpath(filename)
                    output_json_file = output_folder_path.joinpath(detection_mode).joinpath(lib_name).joinpath(filename)
                    check_prediction_sa(instance_name, df, query_llm_tokenizer, query_llm_model, input_json_file, output_json_file, if_reverse=if_reverse)

# statistics summarize over parsed results
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_majority_vote(prediction_results):
    if not prediction_results:
        return ""
    counter = Counter(prediction_results)
    max_count = max(counter.values())
    # return the first item with frequency == max_count
    for item, count in counter.items():
        if count == max_count:
            return str(item)
    return ""

def check_outputs(model_name, llm_judge_name, lib_names=None, if_reverse=False):
    if if_reverse:
        if config.current_task == "result parsing llm diagnosis only":
            output_parsed_path = Path(str(config.path_res)+f"_diagnosis_reversed_{llm_judge_name}")
        else:
            output_parsed_path = Path(str(config.path_res)+f"_reversed_{llm_judge_name}")
    else:
        if config.current_task == "result parsing llm diagnosis only":
            output_parsed_path = Path(str(config.path_res)+f"_diagnosis_{llm_judge_name}")
        else:
            output_parsed_path = Path(str(config.path_res)+f"_{llm_judge_name}")
    final_res = {}
    for detection_mode in ["code", "runinfo"]:
        folder_name_llm = f"crash_detection_{model_name}_{detection_mode}"
        if "runinfo" in detection_mode:
            # get all folders start with folder_path_llm
            detection_mode_folders = [folder for folder in os.listdir(output_parsed_path) if folder.startswith(folder_name_llm)]
        else:
            detection_mode_folders = [folder_name_llm]
        for folder_name in detection_mode_folders:
            folder_path_llm = output_parsed_path.joinpath(folder_name)
            res = {}
            for filename in os.listdir(folder_path_llm):
                if filename.endswith('.json') and ((lib_names is None) or (filename.split('_')[0] in lib_names)):
                    file_path_llm = os.path.join(folder_path_llm, filename)
                    llm_predicts_res = load_json(file_path_llm)
                    # for any value that is more than one line, check if the last line contain one of the list items
                    for i in range(len(llm_predicts_res)):
                        lines = llm_predicts_res[i].splitlines()
                        if len(lines) > 1:
                            # original_response = llm_predicts_res[i]
                            last_line = lines[-1].strip()
                            for keyword in ["partially correct", "reasoning wrong", "correct", "wrong"]:
                                if keyword in last_line.lower():
                                    llm_predicts_res[i] = keyword
                                    # print(f"Normalized multi-line response '{original_response}' to: {keyword}")
                                    break
                    res_majority = check_majority_vote(llm_predicts_res)
                    res[filename[:-len(".json")]] = res_majority
                else:
                    print(f"Skipping file {filename} in {folder_path_llm} as it is not .json format.")
                # elif os.path.isdir(os.path.join(folder_path_llm, filename)) and (((lib_names is None) and (filename=="other")) or (filename in lib_names)):
                #     for sub_filename in os.listdir(os.path.join(folder_path_llm, filename)):
                #         if sub_filename.endswith('.json'):
                #             file_path_llm = os.path.join(folder_path_llm, filename, sub_filename)
                #             llm_predicts_res = load_json(file_path_llm)
                #             res_majority = check_majority_vote(llm_predicts_res)
                #             res[sub_filename[:-len(".json")]] = res_majority
        
            final_res[folder_name] = res
    return final_res

def sort_key(instance: str):
    # Example: "sklearn_1_fixed"
    parts = instance.split("_")
    library = parts[0]                     # 'sklearn'
    number = int(parts[1])                 # 1
    status = parts[2]                      # 'fixed' or 'reproduced'

    # Define order for libraries
    library_order = {
        "tensorflow": 0, "torch": 1, "sklearn": 2, "pandas": 3, "numpy": 4,
        "matplotlib": 5, "seaborn": 6, "lightgbm": 7, "statsmodels": 8,
        "torchvision": 9, "NBspecific": 10
    }
    
    # Define order for status
    status_order = {"fixed": 0, "reproduced": 1}

    return (library_order.get(library, 99), status_order.get(status, 99), number)

def flatten_results_to_table(result_list):
    # structure: (model_task, instance) -> val(str)
    values = {}
    for model_task, entry in result_list.items():
        for instance, val in entry.items():
            key = (model_task, instance)
            values[key] = val

    # Pivot into wide format
    data = {}
    for (model_task, instance), val in values.items():
        data.setdefault(instance, {})[model_task] = val

    df = pd.DataFrame.from_dict(data, orient='index').reset_index().rename(columns={'index': 'instance'})
    # sort by fixed/reproduced then number
    df = df.sort_values(by="instance", key=lambda col: col.map(sort_key)) #.sort_values(by=["model_task", "instance"],).reset_index(drop=True)
    return df

def check_all_parsed_results(model_names, llm_judge_name, lib_names=None, if_reverse=False, if_variants=True):
    res = {}
    for model_name in model_names:
        res = res | check_outputs(model_name, llm_judge_name, lib_names=lib_names, if_reverse=if_reverse)
    res_df = flatten_results_to_table(res)
    # print("Total non-empty values (including instance):", res_df[res_df.columns[1:]].notna().sum().sum())
    # Remove brackets from all columns except the first
    for col in res_df.columns[1:]:
        res_df[col] = res_df[col].map(
            lambda x: str(x).replace("[", "").replace("]", "").strip("'").lower() if isinstance(x, str) else x
        )
    if if_variants:
        # empty values replaced by values from the "crash_detection_{modelname}_code" column
        for row in res_df.index:
            for col in res_df.columns[1:]:
                if "runinfo" in col:
                    prefix = col.split("runinfo")[0]
                    # if all three runinfo_s_v, _r_v, and _s_r are empty, then overwrite all runinfo columns with code column
                    runinfo_cols = [f"{prefix}runinfo_s_v", f"{prefix}runinfo_r_v", f"{prefix}runinfo_s_r"]
                    if all((rc in res_df.columns) and (pd.isna(res_df.loc[row, rc]) or not str(res_df.loc[row, rc]).strip()) for rc in runinfo_cols):
                        model_name = col.split("_")[2]
                        code_col = f"crash_detection_{model_name}_code"
                        if code_col in res_df.columns and pd.notna(res_df.loc[row, code_col]) and res_df.loc[row, code_col]:
                            for rc in res_df.columns:
                                if (rc.startswith(prefix)) and ("runinfo" in rc):
                                    res_df.loc[row, rc] = res_df.loc[row, code_col]
                    elif (pd.isna(res_df.loc[row, col]) or not str(res_df.loc[row, col]).strip()):
                        model_name = col.split("_")[2]
                        code_col = f"crash_detection_{model_name}_code"
                        if code_col in res_df.columns and pd.notna(res_df.loc[row, code_col]) and res_df.loc[row, code_col]:
                            res_df.loc[row, col] = res_df.loc[row, code_col]
    if if_reverse:
        if config.current_task == "result parsing llm diagnosis only":
            output_file = Path(str(config.path_res)+f"_diagnosis_reversed_{llm_judge_name}").joinpath(f"results_parsed_diagnosis_reversed_{llm_judge_name}.xlsx")
        else:
            output_file = Path(str(config.path_res)+f"_reversed_{llm_judge_name}").joinpath(f"results_parsed_reversed_{llm_judge_name}.xlsx")
    else:
        if config.current_task == "result parsing llm diagnosis only":
            output_file = Path(str(config.path_res)+f"_diagnosis_{llm_judge_name}").joinpath(f"results_parsed_diagnosis_{llm_judge_name}.xlsx")
        else:
            output_file = Path(str(config.path_res)+f"_{llm_judge_name}").joinpath(f"results_parsed_{llm_judge_name}.xlsx")
    res_df.to_excel(output_file, index=False, engine="openpyxl")

# def aggregate_parsed_result_summaries(name_parsed_file = "results_parsed_summary_llms"):
#     dataframes = []
    
#     # Load all 5 files
#     for i in range(5):
#         df = pd.read_excel(f"llms/llms_outputs/{name_parsed_file}_{i+1}.xlsx", engine="openpyxl")
#         dataframes.append(df)
    
#     if not dataframes:
#         print("No files found to aggregate")
#         return
    
#     # Start with the first dataframe as base
#     final_df = dataframes[0].copy()
#     different_cells_count = 0
#     total_cells_processed = 0
    
#     # For each cell (except the first column which should be instance names), 
#     # find the majority value across all 5 files
#     for col in final_df.columns[1:]:  # Skip first column (instance names)
#         for idx in final_df.index:
#             # Collect values from all 5 files for this cell
#             values = []
#             for df in dataframes:
#                 if idx < len(df) and col in df.columns:
#                     val = df.iloc[idx][col]
#                     if pd.notna(val): 
#                         values.append(str(val).strip())
            
#             # Check if all values are the same
#             if values:
#                 total_cells_processed += 1
#                 unique_values = set(values)
#                 if len(unique_values) > 1:
#                     different_cells_count += 1
#                     print(f"Difference at row {idx}, col '{col}': {values}")
                
#                 # Find majority value
#                 majority_val = check_majority_vote(values)
#                 final_df.iloc[idx, final_df.columns.get_loc(col)] = majority_val
    
#     # Print summary
#     print(f"Total cells processed: {total_cells_processed}")
#     print(f"Cells with different values: {different_cells_count}")
   
#     # Save the aggregated result
#     output_file = f"llms/llms_outputs/{name_parsed_file}.xlsx"
#     final_df.to_excel(output_file, index=False, engine="openpyxl")
#     print(f"\nSaved aggregated results to {output_file}")
    
#     return final_df

def check_output_sa(sa_name, lib_names=None): #, df=None
    final_res = {}
    for task in ["crash_detection_code", "crash_detection_code_runinfo"]:
        res = {}
        for lib_name in lib_names:
            # folder_path_raw_res = config.path_input.joinpath(sa_name).joinpath(task).joinpath(lib_name)
            folder_path_parsed_res = config.path_res.joinpath(sa_name).joinpath(task).joinpath(lib_name)
            for filename in os.listdir(folder_path_parsed_res):
                if filename.endswith('.json'):
                    folder_path_parsed_res_instance = os.path.join(folder_path_parsed_res, filename)
                    sa_predict_res = load_json(folder_path_parsed_res_instance)
                    # Check if all values are the same
                    if sa_predict_res:
                        unique_values = set(sa_predict_res)
                        if len(unique_values) > 1:
                            print(f"Difference for {folder_path_parsed_res_instance}: {sa_predict_res}")
                    sa_predict_res = check_majority_vote(sa_predict_res)
                    res[filename[:-len(".json")]] = str(sa_predict_res)
        final_res['_'.join([sa_name, task])] = res
    return final_res

def check_all_parsed_results_sa(sa_names, output_file, lib_names=None):
    # df = load_ground_truth() # to mark LLM-evaluated ressults
    res = {}
    for sa_name in sa_names:
        res = res | check_output_sa(sa_name, lib_names) #, df
    res_df = flatten_results_to_table(res)
    # Remove brackets from all columns except the first
    for col in res_df.columns[1:]:
        res_df[col] = res_df[col].map(
            lambda x: str(x).replace("[", "").replace("]", "").replace("'", "").lower() if isinstance(x, str) else x
        )
    res_df.to_excel(output_file, index=False, engine="openpyxl")








#------------------auto evaluate part of the results--------------------------------
from pathlib import Path

def check_prediction_auto(instance_name, input_json_file, check_reasoning_flag=False):
    mode = "LLM"
    if check_reasoning_flag:
        field_value = "check reasoning" # will be manually validated into: "partially correct", "reasoning wrong", "correct"
    else:
        field_value = "correct"
    # if "pylint" in str(input_json_file) or "pyright" in str(input_json_file):
    #     mode = "SA"
    predictions = load_json(input_json_file)
    if mode =="LLM":
        parsed_ress = []
        for i, pred_res in enumerate(predictions):
            if isinstance(pred_res, dict) and ("detection" in pred_res):
                if pred_res["detection"] == False and ("fixed" in instance_name):
                    parsed_ress.append("correct")
                elif pred_res["detection"] == False and ("reproduced" in instance_name):
                    parsed_ress.append("wrong")
                elif pred_res["detection"] == True and ("fixed" in instance_name):
                    parsed_ress.append("wrong")
                elif pred_res["detection"] == True and ("reproduced" in instance_name):
                    parsed_ress.append(field_value)
                else:
                    print(f"LLM - Cannot determine detection result for {input_json_file}")
                    parsed_ress.append("cannot verify")
            else:
                print(f"LLM - 'detection' field not found in {input_json_file}: {i}")
                parsed_ress.append("cannot verify")
        return check_majority_vote(parsed_ress)
    
    # # static analysis tools
    # if (len(predictions) <= 0) and ("fixed" in instance_name):
    #     return "correct"
    # if (len(predictions) <= 0) and ("reproduced" in instance_name):
    #     return "wrong"
    # if (len(predictions) > 0) and ("fixed" in instance_name):
    #     return "wrong"
    # if (len(predictions) > 0) and ("reproduced" in instance_name):
    #     return field_value
    print(f"SA - Cannot automatically determine detection result for instance {instance_name}")
    return "cannot verify"

def check_all_predictions_auto(check_reasoning_flag=False, if_save_excel=True, if_variants=True):
    final_res = {}

    for model_name in ["gemini", "qwen", "gpt5"]: #, "pylint", "pyright"
        # if model_name in ["pylint", "pyright"]:
        #     folder_path_output = Path("sas/sas_outputs/results_raw").joinpath(model_name)
        # else:
        #     folder_path_output = Path("llms/llms_outputs/results_raw").joinpath(model_name)
        if if_variants:
            detection_modes = ["code", "runinfo", "runinfo_full_doc", "runinfo_r_v", "runinfo_s_r", "runinfo_s_v"]
        else:
            detection_modes = ["code", "runinfo"]
        for detection_mode in detection_modes:
            res = {}  # Move res dictionary inside the detection_mode loop
            folder_path = Path(f"llms/llms_outputs/results_raw/crash_detection_{model_name}_{detection_mode}")
            # folder_path = folder_path_output.joinpath(detection_mode)
            
            # Check if folder exists before processing
            if not folder_path.exists():
                print(f"Warning: Folder {folder_path} does not exist, skipping...")
                continue
                
            for filename in os.listdir(folder_path):
                if (filename.endswith('.json')):
                    instance_name = filename[:-len(".json")]
                    input_json_file = folder_path.joinpath(filename)
                    res_auto = check_prediction_auto(instance_name, input_json_file, check_reasoning_flag=check_reasoning_flag)
                    res[instance_name] = res_auto
                else:
                    print(f"Skipping file {filename} in {folder_path} as it is not .json format.")
                # elif os.path.isdir(os.path.join(folder_path, filename)):
                #     for sub_filename in os.listdir(os.path.join(folder_path, filename)):
                #         if sub_filename.endswith('.json'):
                #             instance_name = sub_filename[:-len(".json")]
                #             input_json_file = folder_path.joinpath(filename).joinpath(sub_filename)
                #             res_auto = check_prediction_auto(instance_name, input_json_file, check_reasoning_flag=check_reasoning_flag)
                #             df.loc[df["instance"] == instance_name, f"{model_name}_{detection_mode}"] = res_auto
            final_res[f"crash_detection_{model_name}_{detection_mode}"] = res
    res_df = flatten_results_to_table(final_res)
    if if_variants:
        # empty values replaced by values from the "crash_detection_{modelname}_code" column
        for row in res_df.index:
            for col in res_df.columns[1:]:
                if "runinfo" in col:
                    prefix = col.split("runinfo")[0]
                    # if all three runinfo_s_v, _r_v, and _s_r are empty, then overwrite all runinfo columns with code column
                    runinfo_cols = [f"{prefix}runinfo_s_v", f"{prefix}runinfo_r_v", f"{prefix}runinfo_s_r"]
                    if all((rc in res_df.columns) and (pd.isna(res_df.loc[row, rc]) or not str(res_df.loc[row, rc]).strip()) for rc in runinfo_cols):
                        model_name = col.split("_")[2]
                        code_col = f"crash_detection_{model_name}_code"
                        if code_col in res_df.columns and pd.notna(res_df.loc[row, code_col]) and res_df.loc[row, code_col]:
                            for rc in res_df.columns:
                                if (rc.startswith(prefix)) and ("runinfo" in rc):
                                    res_df.loc[row, rc] = res_df.loc[row, code_col]
                    elif (pd.isna(res_df.loc[row, col]) or not str(res_df.loc[row, col]).strip()):
                        model_name = col.split("_")[2]
                        code_col = f"crash_detection_{model_name}_code"
                        if code_col in res_df.columns and pd.notna(res_df.loc[row, code_col]) and res_df.loc[row, code_col]:
                            res_df.loc[row, col] = res_df.loc[row, code_col]
    if if_save_excel:
        res_df.to_excel(Path("llms/llms_outputs/results_raw/results_parsed_detection_only.xlsx"), index=False, engine="openpyxl")
        return
    return res_df

# def smart_check_all_parsed_results():
#     final_res = {}
#     n = 0
#     n_total = 0
#     for model_name in ["gemini", "qwen", "gpt5"]:
#         for detection_mode in ["code", "runinfo", "runinfo_full_doc", "runinfo_r_v", "runinfo_s_r", "runinfo_s_v"]:
#             res = {}
#             folder_path = Path(f"llms/llms_outputs/results_raw/crash_detection_{model_name}_{detection_mode}")
                
#             for filename in os.listdir(folder_path):
#                 if (filename.endswith('.json')):
#                     n_total += 1
#                     instance_name = filename[:-len(".json")]
#                     input_json_file = folder_path.joinpath(filename)
#                     res_auto = check_prediction_auto(instance_name, input_json_file, check_reasoning_flag=True)
#                     if res_auto == "check reasoning":
#                         n += 1
#                         folder_path_llm = config.path_res.joinpath(f"crash_detection_{model_name}_{detection_mode}")
#                         file_path_llm = os.path.join(folder_path_llm, filename)
#                         llm_predicts_res = load_json(file_path_llm)
#                         # for any value that is more than one line, check if the last line contain one of the list items
#                         for i in range(5):
#                             lines = llm_predicts_res[i].splitlines()
#                             if len(lines) > 1:
#                                 # original_response = llm_predicts_res[i]
#                                 last_line = lines[-1].strip()
#                                 for keyword in ["partially correct", "reasoning wrong", "correct", "wrong"]:
#                                     if keyword in last_line.lower():
#                                         llm_predicts_res[i] = keyword
#                                         # print(f"Normalized multi-line response '{original_response}' to: {keyword}")
#                                         break
#                         res[instance_name] = check_majority_vote(llm_predicts_res)
#                     else:
#                         res[instance_name] = res_auto
#                 else:
#                     print(f"Skipping file {filename} in {folder_path} as it is not .json format.")
#             final_res[f"crash_detection_{model_name}_{detection_mode}"] = res
#     print(f"Total instances needing reasoning check: {n}/{n_total}")
#     res_df = flatten_results_to_table(final_res)
#     # Remove brackets from all columns except the first
#     for col in res_df.columns[1:]:
#         res_df[col] = res_df[col].map(
#             lambda x: str(x).replace("[", "").replace("]", "").strip("'").lower() if isinstance(x, str) else x
#         )
#     # empty values replaced by values from the "crash_detection_{modelname}_code" column
#     for row in res_df.index:
#         for col in res_df.columns[1:]:
#             if "runinfo" in col:
#                 prefix = col.split("runinfo")[0]
#                 # if all three runinfo_s_v, _r_v, and _s_r are empty, then overwrite all runinfo columns with code column
#                 runinfo_cols = [f"{prefix}runinfo_s_v", f"{prefix}runinfo_r_v", f"{prefix}runinfo_s_r"]
#                 if all((rc in res_df.columns) and (pd.isna(res_df.loc[row, rc]) or not str(res_df.loc[row, rc]).strip()) for rc in runinfo_cols):
#                     model_name = col.split("_")[2]
#                     code_col = f"crash_detection_{model_name}_code"
#                     if code_col in res_df.columns and pd.notna(res_df.loc[row, code_col]) and res_df.loc[row, code_col]:
#                         for rc in res_df.columns:
#                             if (rc.startswith(prefix)) and ("runinfo" in rc):
#                                 res_df.loc[row, rc] = res_df.loc[row, code_col]
#                 elif (pd.isna(res_df.loc[row, col]) or not str(res_df.loc[row, col]).strip()):
#                     model_name = col.split("_")[2]
#                     code_col = f"crash_detection_{model_name}_code"
#                     if code_col in res_df.columns and pd.notna(res_df.loc[row, code_col]) and res_df.loc[row, code_col]:
#                         res_df.loc[row, col] = res_df.loc[row, code_col]
#     res_df.to_excel(Path("llms/llms_outputs/results_parsed/results_parsed_smart.xlsx"), index=False, engine="openpyxl")