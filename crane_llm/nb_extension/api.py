"""Kernel-facing entry points.

The browser extension drives the backend by running short snippets from this
module in the user's kernel, so the names here are part of the frontend
contract. Keep them stable.
"""

from __future__ import annotations

import importlib
import json
from typing import Optional

from IPython import get_ipython

from . import assistant as assistant_module
from . import cell_filter as cell_filter_module
from . import extension as extension_module
from . import ipython_hooks as ipython_hooks_module
from . import llm_client as llm_client_module
from . import prompt_builder as prompt_builder_module
from . import runinfo as runinfo_module
from . import session_state as session_state_module
from . import settings as settings_module
from . import ui as ui_module


_INSTANCE: Optional["extension_module.CraneNotebookExtension"] = None


# Dependency order matters. ``importlib.reload`` re-executes a module, so any
# module that did ``from x import name`` rebinds to whatever ``x`` holds at that
# moment. Reloading a dependent before its dependency leaves it holding the old
# function objects, which is why editing runinfo.py or llm_client.py used to
# have no effect until a kernel restart.
_RELOAD_ORDER = (
    "crane_llm.runinfo_parser.summary_rules",
    "crane_llm.runinfo_parser.runtime_summary",
    lambda: cell_filter_module,
    lambda: session_state_module,
    lambda: settings_module,
    lambda: llm_client_module,
    lambda: runinfo_module,
    lambda: prompt_builder_module,
    lambda: ipython_hooks_module,
    lambda: assistant_module,
    lambda: ui_module,
    lambda: extension_module,
)


def _reload_backend_modules() -> None:
    importlib.invalidate_caches()

    for entry in _RELOAD_ORDER:
        module = importlib.import_module(entry) if isinstance(entry, str) else entry()
        importlib.reload(module)


def get_extension(model: Optional[str] = None) -> "extension_module.CraneNotebookExtension":
    """Return the session-wide extension instance, creating it if needed.

    A different ``model`` rebuilds the instance rather than being ignored.
    """

    global _INSTANCE

    if _INSTANCE is not None:
        if model is None or llm_client_module._resolve_model_name(model) == _INSTANCE.model:
            return _INSTANCE
        _dispose_instance()

    _INSTANCE = extension_module.CraneNotebookExtension(model=model)
    _INSTANCE.start_tracking()
    return _INSTANCE


def _dispose_instance() -> None:
    global _INSTANCE

    if _INSTANCE is not None:
        try:
            _INSTANCE.dispose()
        except Exception:
            pass
    _INSTANCE = None


def reload_crane_llm(model: Optional[str] = None) -> "extension_module.CraneNotebookExtension":
    """Reload the notebook backend modules and rebuild the singleton.

    Use this after editing Python backend files in a live kernel session. The
    old instance is disposed first so its ``post_run_cell`` hook is detached
    instead of accumulating with every reload.
    """

    _dispose_instance()
    _reload_backend_modules()
    return get_extension(model=model)


def load_crane_llm(model: Optional[str] = None):
    """Register the notebook helper in the current IPython session."""

    if get_ipython() is None:
        raise RuntimeError("CRANE-LLM notebook helpers require an active IPython session.")
    return get_extension(model=model)


def run_crane_llm(
    source: str,
    model: Optional[str] = None,
    cell_id: str = "active-cell",
    include_runinfo: bool = True,
) -> "extension_module.NotebookExtensionResult":
    """Build the prompt and run the LLM in one call.

    ``include_runinfo=False`` builds the prompt from the executed cells and the
    target cell alone, with no runtime information section.
    """

    extension = get_extension(model=model)
    return extension.run_target_cell(
        source=source,
        shell=get_ipython(),
        cell_id=cell_id,
        render=False,
        include_runinfo=include_runinfo,
    )


def get_live_runinfo_json(target_code: str = "") -> str:
    """Return a JSON string describing the live kernel namespace."""

    runinfo = runinfo_module.collect_live_runinfo(shell=get_ipython(), target_code=target_code)
    return json.dumps(runinfo, ensure_ascii=False, default=str)


def get_prompt(
    source: str = "",
    model: Optional[str] = None,
    cell_id: str = "active-cell",
    include_runinfo: bool = True,
) -> str:
    """Return the current CRANE prompt assembled by the Python backend.

    ``cell_id`` is the notebook's own id for the cell under analysis. It lets
    the prompt builder drop that cell from the executed-cells list when it has
    been run before.
    """

    extension = get_extension(model=model)
    extension.set_target_cell(cell_id=cell_id, source=source)
    return extension.assistant.build_prompt(
        shell=get_ipython(), include_runinfo=include_runinfo
    )


# The frontend reads the payload back off the kernel's stdout. Anything a user
# library happens to print during the call would otherwise be spliced into the
# prompt, so the payload is delimited and everything outside the markers is
# discarded by the caller.
PAYLOAD_BEGIN = "<<<CRANE-LLM:BEGIN>>>"
PAYLOAD_END = "<<<CRANE-LLM:END>>>"


def run_crane_llm_payload(
    source: str = "",
    cell_id: str = "active-cell",
    model: Optional[str] = None,
    include_runinfo: bool = True,
) -> str:
    """Build the prompt, call the LLM, and return a delimited JSON payload.

    Failures are reported inside the payload rather than raised, so the sidebar
    can still show the prompt that was built and a readable error instead of a
    kernel traceback.
    """

    payload = {
        "ok": False,
        "prompt": "",
        "response": "",
        "error": "",
        "include_runinfo": bool(include_runinfo),
    }

    try:
        extension = get_extension(model=model)
        extension.set_target_cell(cell_id=cell_id, source=source)
        payload["prompt"] = extension.assistant.build_prompt(
            shell=get_ipython(), include_runinfo=include_runinfo
        )
    except Exception as exc:
        payload["error"] = f"Prompt building failed. {type(exc).__name__}: {exc}"
        return PAYLOAD_BEGIN + json.dumps(payload, ensure_ascii=False) + PAYLOAD_END

    try:
        payload["response"] = extension.assistant.call_llm(
            payload["prompt"], include_runinfo=include_runinfo
        )
        payload["ok"] = True
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"

    return PAYLOAD_BEGIN + json.dumps(payload, ensure_ascii=False) + PAYLOAD_END


def run_prompt(prompt: str, model: Optional[str] = None, include_runinfo: bool = True) -> str:
    """Run a single prompt through the LLM without rebuilding notebook state."""

    client = llm_client_module.default_client(
        model=model, include_runinfo=include_runinfo
    )
    return client.run(prompt)
