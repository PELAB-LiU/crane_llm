"""Shared runtime-information summarisation.

Both the offline notebook extractor (``notebook_runtime_extractor``) and the
live notebook extension (``nb_extension.runinfo``) need to answer the same
three questions:

1. which names does a target cell depend on,
2. how do we summarise one runtime value,
3. how do we turn (1) + a namespace into the runinfo mapping used in prompts.

Keeping a single implementation here avoids the two copies drifting apart.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import config as crane_config

from . import summary_rules
from .dependency_visitor import DependencyVisitor


Dependencies = Set[str]
Attributes = Set[Tuple[str, str]]


# --- summarisation rules -------------------------------------------------

_RULES_CACHE: Optional[list] = None
_RULES_CACHE_STAMP: Optional[float] = None


def _config_stamp() -> Optional[float]:
    try:
        return os.path.getmtime(crane_config.sum_rule_config_path)
    except OSError:
        return None


def get_summarize_rules(force_reload: bool = False):
    """Return the enabled ``summarize_*`` rule functions.

    The result is cached and invalidated when ``summarize_config.json`` changes
    on disk. The previous implementations re-read and re-parsed that file once
    per summarised variable, which is wasteful in an interactive session.
    """

    global _RULES_CACHE, _RULES_CACHE_STAMP

    stamp = _config_stamp()
    if not force_reload and _RULES_CACHE is not None and stamp == _RULES_CACHE_STAMP:
        return _RULES_CACHE

    with open(crane_config.sum_rule_config_path) as handle:
        runtime_config = json.load(handle)

    enabled_rules = set(runtime_config.get("enabled_rules", []))
    funcs = [
        obj
        for name, obj in inspect.getmembers(summary_rules)
        if inspect.isfunction(obj) and name in enabled_rules
    ]

    _RULES_CACHE = funcs
    _RULES_CACHE_STAMP = stamp
    return funcs


def summarize_variable(
    val: Any,
    namespace: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarise a single runtime value using the enabled rules."""

    summary: Dict[str, Any] = {"type": str(type(val))}
    rules = get_summarize_rules()

    callable_rule = next((rule for rule in rules if rule.__name__ == "summarize_callable"), None)
    if callable_rule is not None:
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


# --- dependency extraction ----------------------------------------------


def _parse_source(code: str, shell: Any = None) -> Optional[ast.AST]:
    """Parse cell source, tolerating IPython syntax and incomplete code.

    Notebook cells routinely contain magics (``%matplotlib inline``, ``%%time``)
    and shell escapes (``!pip install ...``), none of which are valid Python.
    When a plain parse fails we retry through IPython's input transformer,
    which rewrites those forms into ordinary Python. If that also fails the
    cell is genuinely unparseable and the caller gets no dependencies rather
    than an exception.
    """

    try:
        return ast.parse(code)
    except SyntaxError:
        pass

    transformer = getattr(shell, "input_transformer_manager", None)
    if transformer is not None:
        try:
            return ast.parse(transformer.transform_cell(code))
        except Exception:
            return None

    return None


def extract_dependencies(code: str, shell: Any = None) -> Tuple[Dependencies, Attributes]:
    """Return (global names, (base, attribute) pairs) used by ``code``."""

    if not code or not code.strip():
        return set(), set()

    tree = _parse_source(code, shell=shell)
    if tree is None:
        return set(), set()

    visitor = DependencyVisitor()
    visitor.visit(tree)
    return visitor.global_vars, visitor.attr_accesses.union(visitor.method_calls)


# --- runinfo collection --------------------------------------------------


def collect_runtime_info(
    namespace: Dict[str, Any],
    dependencies: Iterable[str],
    attributes: Iterable[Tuple[str, str]],
) -> Dict[str, Dict[str, Any]]:
    """Summarise the parts of ``namespace`` the target cell actually touches.

    Every value is summarised defensively. A live kernel namespace is full of
    third-party objects whose properties can raise, so one bad object must not
    take down the whole prompt.
    """

    relevant: Dict[str, Dict[str, Any]] = {}

    for var in dependencies:
        if var not in namespace:
            continue
        try:
            relevant[var] = summarize_variable(namespace[var], namespace=namespace, name=var)
        except Exception:
            continue

    for base_name, attr in attributes:
        if base_name not in namespace:
            continue
        try:
            obj = namespace[base_name]
            if not hasattr(obj, attr):
                continue
            method = getattr(obj, attr)
            summary = summarize_variable(method, namespace=namespace, name=attr)
        except Exception:
            continue
        relevant[f"__method__{base_name}_{attr}"] = summary

    return relevant
