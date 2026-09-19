import copy
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console

from runinfo_parser import preprocess_notebook
from runinfo_parser import utils
from .cell_executor import IPythonExecutor
from .runinfo_tracker import RuninfoTracker
# Summarisation is shared with the notebook extension; see runtime_summary.
# get_summarize_rules is re-exported because it used to be defined here.
from .runtime_summary import (
    collect_runtime_info,
    extract_dependencies,
    get_summarize_rules,
    summarize_variable,
)

console = Console(force_terminal=True)

@contextmanager
def temporarily_change_dir(target_dir):
    original_dir = os.getcwd()
    try:
        os.chdir(target_dir)
        yield
    finally:
        os.chdir(original_dir)

class NotebookRuntimeExtractor:
    def __init__(self, path: Path, target_cell_id: int = 0):
        self.namespace = {} # runtime info needed
        self.name_origin_map = {} # where does runinfo come from
        self.notebook_dir = os.path.dirname(path)

        # processed_nb: {"executed": [{"execution_count": exec_count, "code_cell_id": code_cell_count, "code": cell.source}], 
        # "target": {"code_cell_id": code_cell_count, "code": cell.source}

        if target_cell_id == 0:
            if "reproduced" in path.name: # buggy
                self.processed_nb= preprocess_notebook.preprocess_buggy_notebook_auto_executed_code_cells(path)
                self.id_name = os.path.basename(path).replace("_reproduced.ipynb","")
            else: # fixed
                self.id_name = os.path.basename(path).replace("_fixed.ipynb","")
                buggy_path = os.path.join(self.notebook_dir, path.name.replace("fixed", "reproduced"))
                self.processed_nb = preprocess_notebook.preprocess_fixed_notebook_auto_executed_code_cells(buggy_path, path)
            self.target_code = self.processed_nb["target"]["code"]
        # todo: take any notebook with a given target cell

        self.runinfo = None
    
    def run_prior_cells(self):
        import time

        """run up to target cell, then record execution time."""
        with temporarily_change_dir(self.notebook_dir):
            try:
                time_start = time.time()
                self.execute_code_cells()
                time_end = time.time()
                exec_time = time_end - time_start
                return exec_time
            except Exception as e:
                console.print(f"[Error] Execution failed: {e}")
                return -1

    def extract(self):
        """Main entry: run up to target cell, then analyze dependencies and extract runtime info."""
        with temporarily_change_dir(self.notebook_dir):
            self.execute_code_cells()
            deps, attrs = self.extract_target_cell_dependencies()
            self.runinfo = self.collect_runtime_info(deps, attrs)

    def get_processed_nb(self):
        # return with all comments removed from the code cells
        res = copy.deepcopy(self.processed_nb)
        for exec_item in res["executed"]:
            exec_item["code"] = utils.remove_comments(exec_item["code"])
        res["target"]["code"] = utils.remove_comments(res["target"]["code"])
        return res

    def get_runinfo_with_source(self) -> Dict[str, Dict[str, Any]]:
        res = {}
        for name in self.runinfo:
            res[name] = self.runinfo[name]
            if name in self.name_origin_map:
                res[name]["execution_cell_source"] = self.name_origin_map[name]
        return res

    def get_runinfo(self) -> Dict[str, Dict[str, Any]]:
        return self.runinfo
    
    def get_runinfo_source(self) -> Dict[str, Dict[str, Any]]:
        res = {}
        for name in self.runinfo:
            if name in self.name_origin_map:
                res[name] = self.name_origin_map[name]
        return res

    def execute_code_cells(self) -> None:
        """Execute all code cells in the execute list"""
        executor = IPythonExecutor()
        # Sort executed cells by execution count
        executed = sorted(self.processed_nb["executed"], key=lambda cell: cell["execution_count"])
        for executed_item in executed:
            cell_code_exec = executed_item["code"]
            # console.print(executed_item["code_cell_id"], executed_item["execution_count"],type(executed_item["execution_count"]))
            tracker = RuninfoTracker()
            try:
                for name, lineno in tracker.get_definitions(cell_code_exec):
                    self.name_origin_map[name] = {"cellno": executed_item["code_cell_id"], "lineno": lineno}
            except Exception as e:
                console.print(f"[Warning] Runinfo tracking failed in cell {executed_item['code_cell_id']}: {e}")
            try:
                # exec(cell_code_exec, self.namespace)
                executor.run_cell(cell_code_exec)
                self.namespace = executor.namespace
                for name, src in utils.get_function_sources(cell_code_exec).items():
                    self.namespace[f'__source__{name}'] = src
            except Exception as e:
                console.print(f"Execution error in {self.id_name} cell {executed_item['code_cell_id']}: {e}. code: {cell_code_exec}")

    def extract_target_cell_dependencies(self) -> List[str]:
        """
        Extract variable and attribute/method dependencies from code.

        Returns:
            variables: Set of variable names used in the code.
            attributes: Set of (base, attr_chain) for attributes and method calls.
                        attr_chain may include chained access like 'layers.0.name'
        """
        return extract_dependencies(self.target_code)

    def summarize_variable(self, val: Any, name: str = None) -> Dict[str, Any]:
        """Summarize variable's type, shape, value info, etc."""
        return summarize_variable(val, namespace=self.namespace, name=name)

    def collect_runtime_info(self, dependencies: List[str], attributes: list) -> Dict[str, Dict[str, Any]]:
        """Collect runtime info based on what are used in target cell."""
        return collect_runtime_info(self.namespace, dependencies, attributes)