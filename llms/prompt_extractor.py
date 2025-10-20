from llms.config_llms import config
import os
import re
from runinfo_parser import notebook_runtime_extractor
import pprint

def generate_prompt(lib_name = "tensorflow", case_names = None, force_regenerate = False):
    # pattern for libname_id
    pattern = re.compile(r'^[A-Za-z]+_\d+$')
    if case_names is None:
        case_names = [
            name for name in os.listdir(config.path_nbs.joinpath(lib_name))
            if os.path.isdir(config.path_nbs.joinpath(lib_name).joinpath(name)) and pattern.match(name)
    ]
    print(f"Generating prompts for {config.current_task}")
    for case_name in case_names:
        for version_name in ["reproduced", "fixed"]:
            path_nb = config.path_nbs.joinpath(lib_name).joinpath(case_name).joinpath(f"{case_name}_{version_name}.ipynb")
            output_path = config.path_input.joinpath(lib_name).joinpath(f"{case_name}_{version_name}.txt")
            if not force_regenerate and output_path.exists():
                print(f"Skipping {case_name}_{version_name} - already exists")
                continue
            print(f"Processing {case_name}_{version_name}")
            extracter = notebook_runtime_extractor.NotebookRuntimeExtractor(path_nb)
            prompt_text = format_for_prompt(extracter)
            if not output_path.parent.exists():
                raise FileNotFoundError(
                    f"Parent directory '{output_path.parent}' does not exist!"
                )
            output_path.write_text(prompt_text, encoding="utf-8")

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

# generate_prompt("numpy")
# generate_prompt_extract_txt(config.path_input_executed_code_runinfo.joinpath("tensorflow"), config.path_input_runinfo.joinpath("tensorflow"))