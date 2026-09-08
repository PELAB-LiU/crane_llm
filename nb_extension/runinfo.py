from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional, Set, Tuple

from IPython import get_ipython

from runinfo_parser.dependency_visitor import DependencyVisitor
from runinfo_parser import summary_rules

import config


config_path = config.sum_rule_config_path


def _extract_target_cell_dependencies(target_code: str) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    if not target_code.strip():
        return set(), set()

    tree = ast.parse(target_code)

    visitor = DependencyVisitor()
    visitor.visit(tree)
    attributes = visitor.attr_accesses.union(visitor.method_calls)
    return visitor.global_vars, attributes


def get_summarize_rules():
    with open(config_path) as handle:
        runtime_config = json.load(handle)

    enabled_rules = set(runtime_config.get("enabled_rules", []))
    funcs = []
    for name, obj in inspect.getmembers(summary_rules):
        if inspect.isfunction(obj) and name in enabled_rules:
            funcs.append(obj)
    return funcs


def summarize_variable(val: Any, namespace: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> Dict[str, Any]:
    summary = {"type": str(type(val))}

    rules = get_summarize_rules()

    callable_rule = next((rule for rule in rules if rule.__name__ == "summarize_callable"), None)
    if callable_rule:
        callable_summary = callable_rule(val, namespace=namespace, name=name)
        if callable_summary is not None:
            summary.update(callable_summary)
            return summary

    for fn in rules:
        if fn.__name__ == "summarize_callable":
            continue
        rule_summary = fn(val)
        if rule_summary:
            summary.update(rule_summary)

    return summary


def collect_live_runinfo(shell: Any = None, target_code: str = "") -> Dict[str, Any]:
    """Collect a compact snapshot of the live kernel namespace.

    This mirrors the offline extractor by analyzing the target cell source,
    resolving dependencies against the current kernel namespace, and returning
    summaries for the referenced variables and methods.
    """

    if shell is None:
        shell = get_ipython()

    if shell is None:
        return {"note": "IPython shell unavailable"}

    user_ns = getattr(shell, "user_ns", {}) or {}

    dependencies, attributes = _extract_target_cell_dependencies(target_code)
    relevant: Dict[str, Dict[str, Any]] = {}

    for name in dependencies:
        if name in user_ns:
            relevant[name] = summarize_variable(user_ns[name], namespace=user_ns, name=name)

    for base_name, attr in attributes:
        if base_name in user_ns:
            obj = user_ns[base_name]
            if hasattr(obj, attr):
                method = getattr(obj, attr)
                method_key = f"__method__{base_name}_{attr}"
                method_summary = summarize_variable(method, namespace=user_ns, name=attr)
                user_ns[method_key] = method_summary
                relevant[method_key] = method_summary

    return relevant


def format_runinfo_for_prompt(runinfo: Dict[str, Any]) -> str:
    return pformat(runinfo)
