from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from IPython.display import Markdown, display

from .assistant import CraneNotebookAssistant
from .ipython_hooks import IPythonSessionTracker
from .session_state import NotebookSessionState
from .ui import CraneNotebookUI


@dataclass
class NotebookExtensionResult:
    prompt: str
    response: str


class CraneNotebookExtension:
    """High-level notebook entry point.

    This class coordinates session tracking, prompt generation, UI rendering,
    and the single-shot LLM call.
    """

    def __init__(self, model: Optional[str] = None):
        self.session_state = NotebookSessionState()
        self.tracker = IPythonSessionTracker(self.session_state)
        self.assistant = CraneNotebookAssistant(model=model, session_state=self.session_state)
        self.ui = CraneNotebookUI()

    def start_tracking(self):
        return self.tracker.register()

    def set_target_cell(self, cell_id: str, source: str, execution_count: Optional[int] = None):
        self.tracker.set_target_cell(cell_id=cell_id, source=source, execution_count=execution_count)

    def run_target_cell(self, source: str, shell=None, cell_id: str = "active-cell", execution_count: Optional[int] = None, render: bool = True) -> NotebookExtensionResult:
        self.set_target_cell(cell_id=cell_id, source=source, execution_count=execution_count)
        return self.run(shell=shell, render=render)

    def run(self, shell=None, render: bool = True) -> NotebookExtensionResult:
        if render:
            self.ui.show()
            self.ui.set_status("parsing runtime information...")
        prompt = self.assistant.build_prompt(shell=shell)
        if render:
            self.ui.set_prompt(prompt)

        if render:
            self.ui.set_status("calling LLM...")
        response = self.assistant.call_llm(prompt)
        if render:
            self.ui.set_status("done")
            self.ui.set_response(response)

        if render:
            display(Markdown(response))
        return NotebookExtensionResult(prompt=prompt, response=response)
