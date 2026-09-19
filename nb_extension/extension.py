from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .assistant import CraneNotebookAssistant
from .ipython_hooks import IPythonSessionTracker
from .session_state import NotebookSessionState


@dataclass
class NotebookExtensionResult:
    prompt: str
    response: str


class CraneNotebookExtension:
    """High-level notebook entry point.

    Coordinates session tracking, prompt generation, optional ipywidgets
    rendering, and the single-shot LLM call.
    """

    def __init__(self, model: Optional[str] = None):
        self.session_state = NotebookSessionState()
        self.tracker = IPythonSessionTracker(self.session_state)
        self.assistant = CraneNotebookAssistant(model=model, session_state=self.session_state)
        self._ui = None

    @property
    def model(self) -> str:
        return self.assistant.model

    @property
    def ui(self):
        """The ipywidgets panel, created on first use.

        Building it eagerly would make ipywidgets a hard requirement of the
        whole backend, including the native frontend path that never renders it.
        """

        if self._ui is None:
            from .ui import CraneNotebookUI

            self._ui = CraneNotebookUI()
        return self._ui

    def start_tracking(self):
        return self.tracker.register()

    def dispose(self) -> None:
        """Detach kernel hooks. Always call this before dropping the instance."""

        self.tracker.dispose()

    def set_target_cell(self, cell_id: str, source: str, execution_count: Optional[int] = None):
        self.tracker.set_target_cell(cell_id=cell_id, source=source, execution_count=execution_count)

    def run_target_cell(
        self,
        source: str,
        shell=None,
        cell_id: str = "active-cell",
        execution_count: Optional[int] = None,
        render: bool = True,
        include_runinfo: bool = True,
    ) -> NotebookExtensionResult:
        self.set_target_cell(cell_id=cell_id, source=source, execution_count=execution_count)
        return self.run(shell=shell, render=render, include_runinfo=include_runinfo)

    def run(
        self,
        shell=None,
        render: bool = True,
        include_runinfo: bool = True,
    ) -> NotebookExtensionResult:
        ui = self.ui if render else None

        if ui is not None:
            ui.show()
            ui.set_status(
                "parsing runtime information..." if include_runinfo else "building prompt..."
            )

        prompt = self.assistant.build_prompt(shell=shell, include_runinfo=include_runinfo)

        if ui is not None:
            ui.set_prompt(prompt)
            ui.set_status("calling LLM...")

        try:
            response = self.assistant.call_llm(prompt, include_runinfo=include_runinfo)
        except Exception as exc:
            if ui is not None:
                ui.set_status("error")
                ui.set_response(f"{type(exc).__name__}: {exc}")
            raise

        if ui is not None:
            ui.set_status("done")
            ui.set_response(response)

        return NotebookExtensionResult(prompt=prompt, response=response)
