from __future__ import annotations

import json
import importlib
from typing import Optional

from IPython import get_ipython

from . import assistant as assistant_module
from . import extension as extension_module
from . import llm_client as llm_client_module
from . import ipython_hooks as ipython_hooks_module
from . import prompt_builder as prompt_builder_module
from . import runinfo as runinfo_module
from . import session_state as session_state_module


def _reload_backend_modules() -> None:
    importlib.reload(session_state_module)
    importlib.reload(ipython_hooks_module)
    importlib.reload(prompt_builder_module)
    importlib.reload(assistant_module)
    importlib.reload(llm_client_module)
    importlib.reload(runinfo_module)
    importlib.reload(extension_module)


def get_extension(model: Optional[str] = None) -> extension_module.CraneNotebookExtension:
    extension_instance = getattr(get_extension, "_instance", None)
    if extension_instance is None:
        extension_instance = extension_module.CraneNotebookExtension(model=model)
        extension_instance.start_tracking()
        setattr(get_extension, "_instance", extension_instance)
    return extension_instance


def reload_crane_llm(model: Optional[str] = None) -> extension_module.CraneNotebookExtension:
    """Reload the notebook backend modules and rebuild the singleton.

    Use this after editing Python backend files in a live kernel session.
    """

    _reload_backend_modules()
    setattr(get_extension, "_instance", None)
    return get_extension(model=model)


def load_crane_llm(model: Optional[str] = None):
    """Register the notebook helper in the current IPython session."""

    shell = get_ipython()
    if shell is None:
        raise RuntimeError("CRANE-LLM notebook helpers require an active IPython session.")
    return get_extension(model=model)


def run_crane_llm(source: str, model: Optional[str] = None) -> extension_module.NotebookExtensionResult:
    """Run CRANE-LLM directly from notebook Python code."""

    extension = get_extension(model=model)
    shell = get_ipython()
    return extension.run_target_cell(source=source, shell=shell, render=False)


def get_live_runinfo_json(target_code: str = "") -> str:
    """Return a JSON string describing the live kernel namespace."""

    return json.dumps(runinfo_module.collect_live_runinfo(shell=get_ipython(), target_code=target_code), ensure_ascii=False, default=str)


def get_prompt(source: str = "", model: Optional[str] = None) -> str:
    """Return the current CRANE prompt assembled by the Python backend."""

    extension = get_extension(model=model)
    shell = get_ipython()
    if source:
        extension.set_target_cell(cell_id="active-cell", source=source)
    return extension.assistant.build_prompt(shell=shell)


def run_prompt(prompt: str, model: Optional[str] = None) -> str:
    """Run a single prompt through the LLM without rebuilding notebook state."""

    client = llm_client_module.default_openai_client(model=model or None)
    return client.run(prompt)
