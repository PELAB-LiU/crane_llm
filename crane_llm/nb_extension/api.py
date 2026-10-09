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
from . import check_helpers as check_helpers_module
from . import check_rules as check_rules_module
from . import checks as checks_module
from . import extension as extension_module
from . import guard as guard_module
from . import ipython_hooks as ipython_hooks_module
from . import llm_client as llm_client_module
from . import prompt_builder as prompt_builder_module
from . import provenance as provenance_module
from . import runinfo as runinfo_module
from . import session_state as session_state_module
from . import settings as settings_module
from . import texts as texts_module
from . import ui as ui_module
from . import verdict as verdict_module


_INSTANCE: Optional["extension_module.CraneNotebookExtension"] = None


# Dependency order matters. ``importlib.reload`` re-executes a module, so any
# module that did ``from x import name`` rebinds to whatever ``x`` holds at that
# moment. Reloading a dependent before its dependency leaves it holding the old
# function objects, which is why editing runinfo.py or llm_client.py used to
# have no effect until a kernel restart.
_RELOAD_ORDER = (
    # First, so that reloading also picks up edits to src/ui_texts.json.
    lambda: texts_module,
    "crane_llm.runinfo_parser.summary_rules",
    "crane_llm.runinfo_parser.runtime_summary",
    lambda: cell_filter_module,
    # The rule registry lives in check_helpers; reloading it empties the
    # registry, and reloading check_rules fills it again.
    lambda: check_helpers_module,
    lambda: check_rules_module,
    lambda: checks_module,
    lambda: verdict_module,
    lambda: provenance_module,
    lambda: session_state_module,
    lambda: settings_module,
    lambda: llm_client_module,
    lambda: runinfo_module,
    lambda: prompt_builder_module,
    lambda: ipython_hooks_module,
    lambda: assistant_module,
    lambda: ui_module,
    lambda: guard_module,
    lambda: extension_module,
)


def _reload_backend_modules() -> None:
    importlib.invalidate_caches()

    for entry in _RELOAD_ORDER:
        module = importlib.import_module(entry) if isinstance(entry, str) else entry()
        importlib.reload(module)


def get_extension() -> "extension_module.CraneNotebookExtension":
    """Return the session-wide extension instance, creating it if needed.

    It holds the record of every cell run since it was created, so it lives
    as long as the kernel. The model is chosen per check, not here.
    """

    global _INSTANCE

    if _INSTANCE is None:
        _INSTANCE = extension_module.CraneNotebookExtension()
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


def reload_crane_llm() -> "extension_module.CraneNotebookExtension":
    """Reload the notebook backend modules and rebuild the singleton.

    Use this after editing Python backend files in a live kernel session. The
    old instance is disposed first so its ``post_run_cell`` hook is detached
    instead of accumulating with every reload.
    """

    _dispose_instance()
    guarded = guard_module.is_enabled(get_ipython())
    _reload_backend_modules()
    extension = get_extension()
    if guarded:
        guard_module.enable(get_ipython())
    return extension


def load_crane_llm():
    """Register the notebook helper in the current IPython session."""

    if get_ipython() is None:
        raise RuntimeError("CRANE-LLM notebook helpers require an active IPython session.")
    return get_extension()


def set_guard(enabled: bool) -> None:
    """Check every cell with the built-in checker before it runs, or stop doing so.

    A cell the checker finds will crash is then not run at all. Lasts until
    the kernel restarts. Reloading the backend keeps the setting, and
    re-installs the guard so that it runs the reloaded code.
    """

    shell = get_ipython()
    if shell is None:
        raise RuntimeError("CRANE-LLM notebook helpers require an active IPython session.")
    # Origins are traced from the cells recorded since the backend loaded.
    get_extension()
    if enabled:
        guard_module.enable(shell)
    else:
        guard_module.disable(shell)


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

    return get_extension().run_target_cell(
        source=source,
        shell=get_ipython(),
        cell_id=cell_id,
        render=False,
        include_runinfo=include_runinfo,
        model=model,
    )


def get_live_runinfo_json(target_code: str = "") -> str:
    """Return a JSON string describing the live kernel namespace."""

    runinfo = runinfo_module.collect_live_runinfo(shell=get_ipython(), target_code=target_code)
    return json.dumps(runinfo, ensure_ascii=False, default=str)


def get_prompt(
    source: str = "",
    cell_id: str = "active-cell",
    include_runinfo: bool = True,
) -> str:
    """Return the current CRANE prompt assembled by the Python backend.

    ``cell_id`` is the notebook's own id for the cell under analysis. It lets
    the prompt builder drop that cell from the executed-cells list when it has
    been run before.
    """

    extension = get_extension()
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
# Prefixes one line of JSON per progress stage, printed before the payload.
STAGE_MARKER = "<<<CRANE-LLM:STAGE>>>"


def run_crane_llm_payload(
    source: str = "",
    cell_id: str = "active-cell",
    model: Optional[str] = None,
    include_runinfo: bool = True,
    notebook_cells_json: str = "",
    use_llm: bool = True,
) -> str:
    """Judge the cell and return a delimited JSON payload.

    The built-in checks run first; the prompt is built and the LLM called only
    when none of them finds a certain crash. ``notebook_cells_json`` lists every
    code cell of the notebook as ``[{"id", "source"}]``, which lets a variable
    that does not exist be traced to the cell that would define it.

    Payload fields: ``verdict`` (tone, label, reasoning, source, certain, model,
    variables, ...), ``origins`` (the cells behind the blamed variables),
    ``prompt`` and ``response`` (both empty when a check answered), ``error``.
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
        "use_llm": bool(use_llm),
        "verdict": None,
        "origins": [],
    }

    try:
        notebook_cells = json.loads(notebook_cells_json) if notebook_cells_json else None
    except ValueError:
        notebook_cells = None

    def progress(stage: str, prompt: str, model_name: str) -> None:
        if stage == assistant_module.STAGE_WAITING:
            payload["prompt"] = prompt
        # Printed as it happens, so the frontend can show each step while the
        # request is still running.
        event = {"stage": stage, "model": model_name, "prompt": prompt}
        print(STAGE_MARKER + json.dumps(event, ensure_ascii=False), flush=True)

    try:
        extension = get_extension()
        extension.set_target_cell(cell_id=cell_id, source=source)
        result = extension.assistant.run(
            shell=get_ipython(),
            include_runinfo=include_runinfo,
            notebook_cells=notebook_cells,
            progress=progress,
            use_llm=use_llm,
            model=model,
        )
    except assistant_module.PromptBuildingError as exc:
        payload["error"] = str(exc)
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    else:
        payload.update(
            ok=True,
            prompt=result.prompt,
            response=result.response,
            verdict=result.verdict.to_json(),
            origins=[origin.to_json() for origin in result.origins],
        )

    return PAYLOAD_BEGIN + json.dumps(payload, ensure_ascii=False, default=str) + PAYLOAD_END


def run_prompt(prompt: str, model: Optional[str] = None, include_runinfo: bool = True) -> str:
    """Run a single prompt through the LLM without rebuilding notebook state."""

    client = llm_client_module.default_client(
        model=model, include_runinfo=include_runinfo
    )
    return client.run(prompt)
