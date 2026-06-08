from llms.config_llms import config
import os
import re
from runinfo_parser import notebook_runtime_extractor
from runinfo_parser.summary_rules import filter_all_category_combinations
import pprint

def runtime_count_prior_cells(lib_name = "tensorflow", case_names = None):
    # pattern for libname_id
    pattern = re.compile(r'^[A-Za-z]+_\d+$')
    if case_names is None:
        case_names = [
            name for name in os.listdir(config.path_nbs.joinpath(lib_name))
            if os.path.isdir(config.path_nbs.joinpath(lib_name).joinpath(name)) and pattern.match(name)
    ]
    res_count = {}
    for case_name in case_names:
        for version_name in ["reproduced"]:
            path_nb = config.path_nbs.joinpath(lib_name).joinpath(case_name).joinpath(f"{case_name}_{version_name}.ipynb")
            print(f"Processing {case_name}_{version_name}")
            extracter = notebook_runtime_extractor.NotebookRuntimeExtractor(path_nb)
            count = extracter.run_prior_cells()
            res_count[f"{case_name}_{version_name}"] = count
    return res_count

def generate_prompt(lib_name = "tensorflow", case_names = None, version_names = None, force_regenerate = False, generate_all_combinations = False):
    # pattern for libname_id
    pattern = re.compile(r'^[A-Za-z]+_\d+$')
    if case_names is None:
        case_names = [
            name for name in os.listdir(config.path_nbs.joinpath(lib_name))
            if os.path.isdir(config.path_nbs.joinpath(lib_name).joinpath(name)) and pattern.match(name)
    ]
    if version_names is None:
        version_names = ["reproduced", "fixed"]
    mode_desc = "all combinations" if generate_all_combinations else "config-based"
    print(f"Generating prompts for {config.current_task} ({mode_desc})")
    
    for case_name in case_names:
        for version_name in version_names:
            path_nb = config.path_nbs.joinpath(lib_name).joinpath(case_name).joinpath(f"{case_name}_{version_name}.ipynb")
            
            print(f"Processing {case_name}_{version_name}")
            extracter = notebook_runtime_extractor.NotebookRuntimeExtractor(path_nb)
            
            # Check if we need to generate all combinations (experimental mode)
            if generate_all_combinations and ("runinfo" in config.current_task):
                # First check if we should skip this case entirely
                if not force_regenerate:
                    # Check if ALL combination files already exist
                    all_combinations = ['full', 's_r', 's_v', 'r_v']
                    already_exist = False
                    for abbrev in all_combinations:
                        folder_suffix = f"_{abbrev}"
                        output_folder = str(config.path_input).replace("executed_code_runinfo", f"executed_code_runinfo{folder_suffix}")
                        output_path = os.path.join(output_folder, lib_name, f"{case_name}_{version_name}.txt")
                        if os.path.exists(output_path):
                            already_exist = True
                            break
                    
                    if already_exist:
                        print(f"Skipping {case_name}_{version_name} - exist in one of the combination files")
                        continue
                
                # Extract the runinfo first
                extracter.extract()
                runinfo = extracter.get_runinfo()
                
                # Generate all category combinations
                filtered_combinations = filter_all_category_combinations(runinfo)
                
                if filtered_combinations:
                    # Generate prompt for each category combination in separate folders
                    for abbrev, filtered_runinfo in filtered_combinations.items():
                        # Create separate folder path for each combination
                        folder_suffix = f"_{abbrev}"
                        output_folder = str(config.path_input).replace("executed_code_runinfo", f"executed_code_runinfo{folder_suffix}")
                        output_path = os.path.join(output_folder, lib_name, f"{case_name}_{version_name}.txt")
                        
                        # Create prompt with filtered runinfo
                        prompt_text = format_for_prompt_with_filtered_runinfo(extracter, filtered_runinfo)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(prompt_text)
                        print(f"Generated {case_name}_{version_name}.txt in folder {folder_suffix[1:]}")
                else:
                    print(f"Warning: No filtered combinations generated for {case_name}_{version_name}")
            else:
                # Original behavior - single output based on config
                output_path = config.path_input.joinpath(lib_name).joinpath(f"{case_name}_{version_name}.txt")
                if not force_regenerate and output_path.exists():
                    # print(f"Skipping {case_name}_{version_name} - already exists")
                    continue
                
                prompt_text = format_for_prompt(extracter)
                if not output_path.parent.exists():
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(prompt_text, encoding="utf-8")
                print(f"Generated {case_name}_{version_name}.txt")

def generate_prompt_extract_txt(src_path, dst_path):
    for file_path in src_path.glob('*.txt'):
        text = file_path.read_text(encoding='utf-8')

        marker_start = "# Current relevent runtime information:\n"
        start_idx = text.find(marker_start)
        if start_idx == -1:
            print(f"Marker start not found in {file_path.name}, skipping...")
            continue
        marker_end = "# Target Cell:\n"
        end_idx = text.find(marker_end)
        if end_idx == -1:
            print(f"Marker end not found in {file_path.name}")
            end_idx = len(text)
        
        extracted = text[start_idx:end_idx]
        target_file = dst_path / file_path.name
        target_file.write_text(extracted, encoding='utf-8')
        print(f"Saved: {target_file}")

def format_for_prompt_with_filtered_runinfo(extracter, filtered_runinfo):
    """
    Format prompt with pre-filtered runinfo for ablation study.
    """
    if not extracter:
        print("Error: extracter is None")
    prompt_text = ""
    
    # Always include executed code cells for runinfo tasks
    processed_nb = extracter.get_processed_nb()
    prompt_text += "# Executed Cells:\n"

    if processed_nb["executed"]:
        # Sort executed cells by execution count
        executed = sorted(processed_nb["executed"], key=lambda cell: cell["execution_count"])
        executed_cells = [cell["code"] for cell in executed]

        for i, code in enumerate(executed_cells, start=1):
            prompt_text += f"## Cell {i}:\n{code}\n\n"
    else:
        prompt_text += "No cell has been executed\n"
    
    # Add the filtered runinfo
    prompt_text += "# Current relevent runtime information:\n"
    prompt_text += pprint.pformat(filtered_runinfo)
    prompt_text += "\n"
        
    # Get the target cell code
    target_cell = processed_nb["target"]["code"]

    prompt_text += "# Target Cell:\n"
    prompt_text += target_cell

    return prompt_text

def format_for_prompt(extracter):
    if not extracter:
        print("Error: extracter is None")
    prompt_text = ""
    if ("executed code cells" in config.current_task):
        processed_nb = extracter.get_processed_nb()
        prompt_text += "# Executed Cells:\n"

        if processed_nb["executed"]:
            # Sort executed cells by execution count
            executed = sorted(processed_nb["executed"], key=lambda cell: cell["execution_count"])
            executed_cells = [cell["code"] for cell in executed]

            for i, code in enumerate(executed_cells, start=1):
                prompt_text += f"## Cell {i}:\n{code}\n\n"
        else:
            prompt_text += "No cell has been executed\n"
    if ("runinfo" in config.current_task):
        extracter.extract()
        runinfo = extracter.get_runinfo_with_source()
        prompt_text += "# Current relevent runtime information:\n"
        prompt_text += pprint.pformat(runinfo)
        prompt_text += "\n"
        
    # Get the target cell code
    target_cell = processed_nb["target"]["code"]

    prompt_text += "# Target Cell:\n"
    prompt_text += target_cell

    return prompt_text
