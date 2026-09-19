from __future__ import annotations

from pprint import pformat
from typing import Any, Dict


from runinfo_parser.runtime_summary import (
    collect_runtime_info,
    extract_dependencies,
    get_summarize_rules,
    summarize_variable,
)

__all__ = [
    "collect_live_runinfo",
    "collect_runtime_info",
    "extract_dependencies",
    "format_runinfo_for_prompt",
    "get_summarize_rules",
    "summarize_variable",
]


def collect_live_runinfo(shell: Any = None, target_code: str = "") -> Dict[str, Any]:
    """Collect a compact snapshot of the live kernel namespace.

    This mirrors the offline extractor: analyse the target cell source, resolve
    the names it uses against the current kernel namespace, and summarise them.

    The kernel namespace is left untouched. An earlier version wrote the method
    summaries back into it, which leaked ``__method__*`` names into the user's
    globals, completions and ``%whos`` output.
    """

    if shell is None:
        from IPython import get_ipython

        shell = get_ipython()

    if shell is None:
        return {"note": "IPython shell unavailable"}

    user_ns = getattr(shell, "user_ns", {}) or {}

    dependencies, attributes = extract_dependencies(target_code, shell=shell)
    return collect_runtime_info(user_ns, dependencies, attributes)


def format_runinfo_for_prompt(runinfo: Dict[str, Any]) -> str:
    return pformat(runinfo)
