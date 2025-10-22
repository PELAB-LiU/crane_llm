import json
import os
import pprint
import pandas as pd
import re
from llms.config_llms import config
from llms import llm_executor
from collections import Counter

# LLM result parsing
def load_ground_truth():
    df = pd.read_excel(config.path_ground_truth, sheet_name="Ground_truth")
    return df

def find_ground_truth(df, instance_name):
    gt = df[df["instance"]==instance_name]
    res = ""
    if len(gt)==1:
        res = f"\nGround truth: {gt.iloc[0]['detection']}. "
        reason = gt.iloc[0]["diagnosis"]
        if pd.notna(reason) and reason: # append reasoning
            res += reason
        elif gt.iloc[0]['detection'] is True: # buggy instance but no reasoning
            print(f"Buggy instance lacks ground truth of reasoning: {instance_name}")
    else:
        print(f"Instance not found in ground truth file: {instance_name}")
    return res

def _normalize_res(res):
    normalized = re.search(r'\[(.*?)\]', str(res))
    if normalized:
        normalized = normalized.group(1).strip().lower()
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

def check_prediction(instance_name, df, query_llm_tokenizer, query_llm_model, input_json_file, output_json_file, if_reverse=False):
    predictions = load_json(input_json_file)
    parsed_ress = []
    # print(f"Predicting {input_json_file}...")
    for pred_res in predictions:
        if not if_reverse:
            user_message = find_ground_truth(df, instance_name) + f"\nPrediction from LLM: {pred_res}\n"
        else:
            user_message = f"\nPrediction from LLM: {pred_res}\n" + find_ground_truth(df, instance_name)
        # print(user_message)
        exec_llm = llm_executor.LLMExecutor(user_message = user_message)
        res = exec_llm.llm_run_huggingface(query_llm_tokenizer, query_llm_model)
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

def check_all_predictions(model_name, query_llm_tokenizer, query_llm_model, lib_names=None, if_reverse=False):
    df = load_ground_truth()
    folder_path_llm = config.path_input.joinpath(model_name)
    for detection_mode in ["crash_detection_code", "crash_detection_code_runinfo"]:
        folder_path_llm = config.path_input.joinpath(model_name).joinpath(detection_mode)
        for filename in os.listdir(folder_path_llm):
            if (filename.endswith('.json')) and ((lib_names is None) or (filename.split('_')[0] in lib_names)):
                instance_name = filename[:-len(".json")]
                input_json_file = folder_path_llm.joinpath(filename)
                output_json_file = config.path_res.joinpath(model_name).joinpath(detection_mode).joinpath(filename)
                check_prediction(instance_name, df, query_llm_tokenizer, query_llm_model, input_json_file, output_json_file, if_reverse=if_reverse)
            elif os.path.isdir(os.path.join(folder_path_llm, filename)) and (((lib_names is None) and (filename=="other")) or (filename in lib_names)):
                for sub_filename in os.listdir(os.path.join(folder_path_llm, filename)):
                    if sub_filename.endswith('.json'):
                        instance_name = sub_filename[:-len(".json")]
                        input_json_file = folder_path_llm.joinpath(filename).joinpath(sub_filename)
                        output_json_file = config.path_res.joinpath(model_name).joinpath(detection_mode).joinpath(filename).joinpath(sub_filename)
                        check_prediction(instance_name, df, query_llm_tokenizer, query_llm_model, input_json_file, output_json_file, if_reverse=if_reverse)

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

def check_outputs(model_name, lib_names=None):
    final_res = {}
    for task in ["crash_detection_code", "crash_detection_code_runinfo"]:
        folder_path_parsed_res = config.path_res.joinpath(model_name).joinpath(task)
        res = {}
        for filename in os.listdir(folder_path_parsed_res):
            if filename.endswith('.json') and ((lib_names is None) or (filename.split('_')[0] in lib_names)):
                file_path_llm = os.path.join(folder_path_parsed_res, filename)
                llm_predicts_res = load_json(file_path_llm)
                res_majority = check_majority_vote(llm_predicts_res)
                res[filename[:-len(".json")]] = res_majority
            elif os.path.isdir(os.path.join(folder_path_parsed_res, filename)) and (((lib_names is None) and (filename=="other")) or (filename in lib_names)):
                for sub_filename in os.listdir(os.path.join(folder_path_parsed_res, filename)):
                    if sub_filename.endswith('.json'):
                        file_path_llm = os.path.join(folder_path_parsed_res, filename, sub_filename)
                        llm_predicts_res = load_json(file_path_llm)
                        res_majority = check_majority_vote(llm_predicts_res)
                        res[sub_filename[:-len(".json")]] = res_majority
    
        final_res['_'.join([model_name, task])] = res
    return final_res

def sort_key(instance: str):
    # Example: "sklearn_1_fixed"
    parts = instance.split("_")
    library = parts[0]                     # 'sklearn'
    number = int(parts[1])                 # 1
    status = parts[2]                      # 'fixed' or 'reproduced'

    # Define order for status
    status_order = {"fixed": 0, "reproduced": 1}

    return (library, status_order.get(status, 99), number)

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

def check_all_parsed_results(model_names, output_file, lib_names=None):
    res = {}
    for model_name in model_names:
        res = res | check_outputs(model_name, lib_names)
    res_df = flatten_results_to_table(res)
    # Remove brackets from all columns except the first
    for col in res_df.columns[1:]:
        res_df[col] = res_df[col].map(
            lambda x: str(x).replace("[", "").replace("]", "").strip("'").lower() if isinstance(x, str) else x
        )
    res_df.to_excel(output_file, index=False, engine="openpyxl")

def aggregate_parsed_result_summaries(name_parsed_file = "results_parsed_summary_llms"):
    dataframes = []
    
    # Load all 5 files
    for i in range(5):
        df = pd.read_excel(f"llms/llms_outputs/{name_parsed_file}_{i+1}.xlsx", engine="openpyxl")
        dataframes.append(df)
    
    if not dataframes:
        print("No files found to aggregate")
        return
    
    # Start with the first dataframe as base
    final_df = dataframes[0].copy()
    different_cells_count = 0
    total_cells_processed = 0
    
    # For each cell (except the first column which should be instance names), 
    # find the majority value across all 5 files
    for col in final_df.columns[1:]:  # Skip first column (instance names)
        for idx in final_df.index:
            # Collect values from all 5 files for this cell
            values = []
            for df in dataframes:
                if idx < len(df) and col in df.columns:
                    val = df.iloc[idx][col]
                    if pd.notna(val): 
                        values.append(str(val).strip())
            
            # Check if all values are the same
            if values:
                total_cells_processed += 1
                unique_values = set(values)
                if len(unique_values) > 1:
                    different_cells_count += 1
                    print(f"Difference at row {idx}, col '{col}': {values}")
                
                # Find majority value
                majority_val = check_majority_vote(values)
                final_df.iloc[idx, final_df.columns.get_loc(col)] = majority_val
    
    # Print summary
    print(f"Total cells processed: {total_cells_processed}")
    print(f"Cells with different values: {different_cells_count}")
   
    # Save the aggregated result
    output_file = f"llms/llms_outputs/{name_parsed_file}.xlsx"
    final_df.to_excel(output_file, index=False, engine="openpyxl")
    print(f"\nSaved aggregated results to {output_file}")
    
    return final_df

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

def load_human_evaluation_file():
    df = pd.read_excel(Path("results/results_human_evaluated.xlsx"), sheet_name="Human_evaluation")
    return df

def check_prediction_auto(instance_name, input_json_file, check_reasoning_flag=False):
    mode = "LLM"
    if check_reasoning_flag:
        field_value = "check reasoning" # will be manually validated into: "partially correct", "reasoning wrong", "correct"
    else:
        field_value = "correct"
    if "pylint" in str(input_json_file) or "pyright" in str(input_json_file):
        mode = "SA"
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
    
    # static analysis tools
    if (len(predictions) <= 0) and ("fixed" in instance_name):
        return "correct"
    if (len(predictions) <= 0) and ("reproduced" in instance_name):
        return "wrong"
    if (len(predictions) > 0) and ("fixed" in instance_name):
        return "wrong"
    if (len(predictions) > 0) and ("reproduced" in instance_name):
        return field_value
    print(f"SA - Cannot automatically determine detection result for instance {instance_name}")
    return "cannot verify"

def check_all_predictions_auto(check_reasoning_flag=False):
    df = load_human_evaluation_file()
    df = df.astype(str)

    for model_name in ["gemini_2_5_flash", "gpt_5", "Qwen_2_5_32B_Instruct", "pylint", "pyright"]:
        if model_name in ["pylint", "pyright"]:
            folder_path_output = Path("sas/sas_outputs/results_raw").joinpath(model_name)
        else:
            folder_path_output = Path("llms/llms_outputs/results_raw").joinpath(model_name)
        for detection_mode in ["crash_detection_code", "crash_detection_code_runinfo"]:
            folder_path = folder_path_output.joinpath(detection_mode)
            for filename in os.listdir(folder_path):
                if (filename.endswith('.json')):
                    instance_name = filename[:-len(".json")]
                    input_json_file = folder_path.joinpath(filename)
                    res_auto = check_prediction_auto(instance_name, input_json_file, check_reasoning_flag=check_reasoning_flag)
                    df.loc[df["instance"] == instance_name, f"{model_name}_{detection_mode}"] = res_auto
                elif os.path.isdir(os.path.join(folder_path, filename)):
                    for sub_filename in os.listdir(os.path.join(folder_path, filename)):
                        if sub_filename.endswith('.json'):
                            instance_name = sub_filename[:-len(".json")]
                            input_json_file = folder_path.joinpath(filename).joinpath(sub_filename)
                            res_auto = check_prediction_auto(instance_name, input_json_file, check_reasoning_flag=check_reasoning_flag)
                            df.loc[df["instance"] == instance_name, f"{model_name}_{detection_mode}"] = res_auto
    df.to_excel(Path("results/results_detection_only.xlsx"), index=False, engine="openpyxl")
