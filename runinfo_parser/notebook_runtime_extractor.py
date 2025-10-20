import ast
from typing import Dict, Any, List, Tuple
from pathlib import Path
import os
from runinfo_parser import preprocess_notebook
from rich.console import Console
from contextlib import contextmanager
import inspect
from runinfo_parser import utils
from .dependency_visitor import DependencyVisitor
from runinfo_parser import summary_rules
from .cell_executor import IPythonExecutor
from .runinfo_tracker import RuninfoTracker
import json
import copy
import config

console = Console(force_terminal=True)
config_path = config.sum_rule_config_path

@contextmanager
def temporarily_change_dir(target_dir):
    original_dir = os.getcwd()
    try:
        os.chdir(target_dir)
        yield
    finally:
        os.chdir(original_dir)

def get_summarize_rules():
    # Load config file
    with open(config_path) as f:
        config = json.load(f)
    enabled_rules = set(config.get("enabled_rules", []))

    # Get all functions from summary_rules starting with 'summarize_' and matching config
    funcs = []
    for name, obj in inspect.getmembers(summary_rules):
        if (inspect.isfunction(obj) and name in enabled_rules):
            funcs.append(obj)
    return funcs

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
        try:
            tree = ast.parse(self.target_code)
        except SyntaxError:
            return [], []
        visitor = DependencyVisitor()
        visitor.visit(tree)
        attributes = visitor.attr_accesses.union(visitor.method_calls)
        return visitor.global_vars, attributes

    def summarize_variable(self, val: Any, name: str = None) -> Dict[str, Any]:
        """Summarize variable's type, shape, value info, etc."""
        summary = {'type': str(type(val))}

        # Load rules dynamically
        rules = get_summarize_rules()

        # Special case for callable: call with extra args
        callable_rule = next((r for r in rules if r.__name__ == 'summarize_callable'), None)
        if callable_rule:
            callable_summary = callable_rule(val, namespace=self.namespace, name=name)
            if callable_summary is not None:
                summary.update(callable_summary)
                return summary

        # Run other rules (skip callable_rule if already applied)
        for fn in rules:
            if fn.__name__ == 'summarize_callable':
                continue
            rule_summary = fn(val)
            if rule_summary:
                summary.update(rule_summary)

        return summary

    def collect_runtime_info(self, dependencies: List[str], attributes: list) -> Dict[str, Dict[str, Any]]:
        """Collect runtime info based on what are used in target cell."""
        relevant = {}
        # variables
        for var in dependencies:
            if var in self.namespace:
                try:
                    relevant[var] = self.summarize_variable(self.namespace[var], name=var)
                except Exception as e:
                    pass
                    # relevant[var] = {'error': str(e)}
        # Summarize methods used in the code
        for base_name, attr in attributes:
            if base_name in self.namespace:
                obj = self.namespace[base_name]
                if hasattr(obj, attr):
                    method = getattr(obj, attr)
                    method_key = f"__method__{base_name}_{attr}"
                    method_summary = self.summarize_variable(method, attr)
                    self.namespace[method_key] = method_summary
                    relevant[method_key] = method_summary
        return relevant