"""Checking each cell before it runs, and stopping it when it would crash.

With the guard on, every cell the user runs goes through the built-in checker
first. When the checker finds a certain crash, the cell is not run at all:
CRANE-LLM shows why, and the cell ends with a ``CrashPrevented`` error instead
of the crash, so nothing in it has executed and a Run All stops there just as
it would have at the crash. Only the checker is used, never the model: it is
local, fast, sends nothing anywhere, and is never wrong about a crash it
reports.

This lives entirely in the kernel, so it works wherever the cell magic does --
Kaggle, Colab, VS Code -- and the JupyterLab switch only turns it on and off.

How it hooks in. IPython applies its AST transformers to a cell after parsing
it and before running it, and only when it actually runs: unlike
``transform_cell``, which the checker and the origin tracker call themselves
to parse a cell, ``transform_ast`` is never called just to read code. A
transformer may refuse a cell by raising ``InputRejected``; IPython then shows
the error and runs nothing. The transformer sees only the parsed code, so the
cell's raw source is taken from the ``pre_run_cell`` event, which fires just
before, and is used only by the very next ``transform_ast`` call.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Optional

from IPython.core.error import InputRejected

from . import checks as checks_module
from .cell_filter import is_internal_helper_cell
from .texts import text


# The guard's output carries its verdict under this type as well as HTML. The
# JupyterLab extension draws it as its own verdict box, with links to the
# origin cells; other frontends do not know the type and show the HTML. Must
# match GUARD_MIME in src/index.ts.
MIME_TYPE = "application/vnd.crane-llm.guard+json"

# A line of its own in a cell that the user wants run whatever the checker says.
_RUN_ANYWAY_RE = re.compile(r"^[ \t]*#[ \t]*crane:[ \t]*run[ \t]*$", re.IGNORECASE | re.MULTILINE)


class CrashPrevented(InputRejected):
    """Raised instead of running a cell the built-in checker found will crash."""

    def __init__(self, reasoning: str):
        super().__init__(text("guard.error", reasoning=reasoning))

    def _render_traceback_(self):
        # IPython shows this instead of a traceback. There is nothing to trace:
        # no line of the cell ran, and the frames would be CRANE-LLM's own.
        return [f"\x1b[0;31m{type(self).__name__}\x1b[0m: {self}"]


class CellGuard(ast.NodeTransformer):
    """The AST transformer that checks each cell before IPython runs it."""

    # How the guard is found again in ``shell.ast_transformers``, also after
    # this module has been reloaded and the class is a different object.
    crane_llm_guard = True

    def __init__(self, shell):
        super().__init__()
        self.shell = shell
        self._next: Optional[tuple] = None

    # --- events ----------------------------------------------------------

    def pre_run_cell(self, info=None) -> None:
        """Remember the cell about to run, if it is one to check."""

        self._next = None
        raw_cell = getattr(info, "raw_cell", None)
        # Requests that do not store history come from frontends and tools,
        # not from the user running a cell.
        if not isinstance(raw_cell, str) or not getattr(info, "store_history", True):
            return
        if is_internal_helper_cell(raw_cell) or raw_cell.lstrip().startswith("%%"):
            return
        if _RUN_ANYWAY_RE.search(raw_cell):
            return
        self._next = (raw_cell, getattr(info, "cell_id", None))

    def post_run_cell(self, result=None) -> None:
        # A cell that failed to parse never reaches transform_ast, and must
        # not leave its source behind for the next execution to be judged by.
        self._next = None

    # --- the check -------------------------------------------------------

    def visit(self, node: Any) -> Any:
        pending, self._next = self._next, None
        if pending is None or not isinstance(node, ast.Module):
            return node
        raw_cell, cell_id = pending
        try:
            stop = self._stop_reason(raw_cell, cell_id)
        except Exception:
            # An exception other than InputRejected would make IPython
            # unregister the guard, and a bug here must never stop a cell.
            stop = None
        if stop is not None:
            raise stop
        return node

    def _stop_reason(self, raw_cell: str, cell_id: Optional[str]) -> Optional[CrashPrevented]:
        namespace = getattr(self.shell, "user_ns", None) or {}
        finding = checks_module.run_checks(raw_cell, namespace, shell=self.shell)
        if finding is None:
            return None

        from . import api
        from .provenance import locate_origins
        from .ui import VerdictView
        from .verdict import verdict_from_finding

        verdict = verdict_from_finding(finding)
        verdict.label = text("guard.verdict_label")
        try:
            origins = locate_origins(
                verdict.variables,
                api.get_extension().session_state.provenance,
                namespace,
                target_cell_id=cell_id,
            )
        except Exception:
            origins = []
        footer = text("guard.footer")
        payload = {
            "verdict": verdict.to_json(),
            "origins": [origin.to_json() for origin in origins],
            "footer": footer,
        }
        try:
            VerdictView(footer=footer).show_once(verdict, origins, {MIME_TYPE: payload})
        except Exception:
            pass
        return CrashPrevented(verdict.reasoning)


# --- switching it on and off ----------------------------------------------


def _registered(shell) -> list:
    return [t for t in getattr(shell, "ast_transformers", []) if getattr(t, "crane_llm_guard", False)]


def enable(shell) -> None:
    """Check every cell before it runs. Enabling twice installs one guard."""

    disable(shell)
    guard = CellGuard(shell)
    shell.ast_transformers.append(guard)
    shell.events.register("pre_run_cell", guard.pre_run_cell)
    shell.events.register("post_run_cell", guard.post_run_cell)


def disable(shell) -> None:
    for guard in _registered(shell):
        shell.ast_transformers.remove(guard)
        for event in ("pre_run_cell", "post_run_cell"):
            try:
                shell.events.unregister(event, getattr(guard, event))
            except ValueError:
                pass


def is_enabled(shell) -> bool:
    return bool(_registered(shell))
