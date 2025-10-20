import os
import re
from typing import Dict, List, Optional, Tuple
from nbformat import NO_CONVERT, read


def parse_traceback(str_traceback):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', str_traceback)

def extract_bug_location_from_cell(cell: dict) -> Optional[Tuple[Optional[int], Optional[str]]]:
    """
    Extract the crashing line number from the error traceback in a code cell's output.
    Returns a 1-based line number and line of code or None if not found.
    """
    if 'outputs' not in cell:
        return None, None
    
    for output in cell['outputs']:
        if output.output_type == 'error':
            traceback_lines = output.get('traceback', [])
            traceback_lines = parse_traceback("\n".join(traceback_lines))
            pattern = r'<ipython-input-(\d+)-[\da-f]+> in <cell line: (\d+)>()'
            match = re.search(pattern, traceback_lines)
            if match:
                line_number = int(match.group(2))
                source_lines = cell.get("source", [])
                if isinstance(source_lines, str):
                    source_lines = source_lines.splitlines()
                if 1 <= line_number <= len(source_lines):
                    return line_number, source_lines[line_number - 1].strip()
    return None, None

def find_buggy_cell_index_and_line(nb_cells: List[dict]) -> Tuple[Optional[int], Optional[int]]:
    """
    Identify the buggy code cell index and the crashing line number.
    """
    code_cell_index = 0
    for cell in nb_cells:
        if cell.cell_type != 'code':
            continue
        for output in cell.get('outputs', []):
            if output.output_type == 'error':
                line = extract_bug_location_from_cell(cell)
                return code_cell_index, line
        code_cell_index += 1
    return None, None

# [for detecting if a target cell crash]
# process reproduced crashing notebooks to: a list of successfully executed code cells, crashing code cell (target)
def preprocess_buggy_notebook_auto_executed_code_cells(nb_path: str):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = read(f, as_version=NO_CONVERT)

    id_name = os.path.basename(nb_path).replace("_reproduced.ipynb","")

    buggy_index, _ = find_buggy_cell_index_and_line(nb.cells)
    if buggy_index is None:
        raise ValueError(f"No error output found in the notebook {id_name}.")
    code_cells = [cell for cell in nb.cells if cell.get("cell_type") == "code"]
    buggy_exec_count = code_cells[buggy_index].get('execution_count')
    
    processed_nb = {"executed": [], "target": None}
    code_cell_count = 0  # Track code cells for logical indexing
    for cell in code_cells:
        exec_count = cell.get('execution_count')
        first_line = cell.source.strip().splitlines()[0] if cell.source.strip() else ""
        # print(code_cell_count, buggy_index, exec_count, buggy_exec_count)
        if ("[reexecute]" in first_line) or ((code_cell_count != buggy_index) and (exec_count is not None) and (exec_count < buggy_exec_count)):
            processed_nb["executed"].append({
                "execution_count": exec_count, 
                "code_cell_id": code_cell_count, 
                "code": cell.source})
        if code_cell_count == buggy_index:
            processed_nb["target"] = {
                "code_cell_id": code_cell_count, 
                "code": cell.source}
        code_cell_count += 1

    if processed_nb["target"] is None:
        print(f"No target cell assigned to {id_name}!")
    if len(processed_nb["executed"])<=0:
        print(f"No executed cell(s) assigned to {id_name}!")

    return processed_nb

# [for detecting if a target cell crash]
# process fixed notebooks to: a list of successfully executed code cells, used-to-crash code cell (target)
def preprocess_fixed_notebook_auto_executed_code_cells(nb_buggy_path: str, nb_fix_path: str):
    with open(nb_buggy_path, 'r', encoding='utf-8') as f:
        nb_buggy = read(f, as_version=NO_CONVERT)
    with open(nb_fix_path, 'r', encoding='utf-8') as f:
        nb_fix = read(f, as_version=NO_CONVERT)

    id_name = os.path.basename(nb_buggy_path).replace("_reproduced.ipynb","")

    processed_nb = {"executed": [], "target": None}
    code_cell_count = 0  # Track code cells for logical indexing

    target_cell_id_buggy, _ = find_buggy_cell_index_and_line(nb_buggy.cells)
    if target_cell_id_buggy is None:
        raise ValueError(f"No error output found in the notebook {id_name}.")
    code_cells = [cell for cell in nb_fix.cells if cell.get("cell_type") == "code"]
    target_exec_count_fixed = code_cells[target_cell_id_buggy].get('execution_count')

    for cell in code_cells:
        exec_count = cell.get('execution_count')
        first_line = cell.source.strip().splitlines()[0] if cell.source.strip() else ""
        if ("[reexecute]" in first_line) or ((code_cell_count != target_cell_id_buggy) and (exec_count is not None) and (exec_count < target_exec_count_fixed)):
            processed_nb["executed"].append({
                "execution_count": exec_count, 
                "code_cell_id": code_cell_count, 
                "code": cell.source})
        if code_cell_count == target_cell_id_buggy:
            processed_nb["target"] = {
                "code_cell_id": code_cell_count, 
                "code": cell.source}
        code_cell_count += 1

    if processed_nb["target"] is None:
        print(f"No target cell assigned to {id_name}!")
    if len(processed_nb["executed"])<=0:
        print(f"No executed cell(s) assigned to {id_name}!")

    return processed_nb


