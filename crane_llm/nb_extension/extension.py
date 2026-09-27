from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from IPython import get_ipython

from .assistant import CraneNotebookAssistant
from .cell_filter import is_internal_helper_cell
from .ipython_hooks import IPythonSessionTracker
from .session_state import NotebookSessionState


@dataclass
class NotebookExtensionResult:
    prompt: str
    response: str


class CraneNotebookExtension:
    """High-level notebook entry point.

    Coordinates session tracking, prompt generation, the magic's cell output,
    and the single-shot LLM call.
    """

    def __init__(self, model: Optional[str] = None):
        self.session_state = NotebookSessionState()
        self.tracker = IPythonSessionTracker(self.session_state)
        self.assistant = CraneNotebookAssistant(model=model, session_state=self.session_state)
        # Verdicts shown by the magic that are not stale yet.
        self._live_views: List = []
        self._stale_shell = None

    @property
    def model(self) -> str:
        return self.assistant.model

    def start_tracking(self):
        registration = self.tracker.register()
        self._register_stale_hook()
        return registration

    def dispose(self) -> None:
        """Detach kernel hooks. Always call this before dropping the instance."""

        self.tracker.dispose()
        if self._stale_shell is not None:
            try:
                self._stale_shell.events.unregister("post_run_cell", self._mark_views_stale)
            except Exception:
                pass
            self._stale_shell = None

    # --- stale marking ---------------------------------------------------

    def _register_stale_hook(self) -> None:
        shell = get_ipython()
        if shell is None or self._stale_shell is not None:
            return
        shell.events.register("post_run_cell", self._mark_views_stale)
        self._stale_shell = shell

    def _mark_views_stale(self, result) -> None:
        """Any user cell that ran may have changed what the verdicts rested on.

        The extension's own cells do not count: a ``%%crane_llm`` cell analyses
        its body without running it, so checking a second cell must not retire
        the first check. That includes the check that has just been shown,
        whose own cell is the one this hook fires for first.
        """

        raw_cell = getattr(getattr(result, "info", None), "raw_cell", None)
        if not isinstance(raw_cell, str) or is_internal_helper_cell(raw_cell):
            return

        views, self._live_views = self._live_views, []
        for view in views:
            view.mark_stale()

    # --- running ---------------------------------------------------------

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
        """Build the prompt and call the model.

        With ``render`` the progress and the verdict are shown as this cell's
        output. Failures are shown there too and then re-raised, so a caller
        can still tell that no prediction was made.
        """

        view = None
        if render:
            from .ui import VerdictView

            view = VerdictView()
            view.show(
                "collecting runtime information..." if include_runinfo else "building the prompt..."
            )

        try:
            prompt = self.assistant.build_prompt(shell=shell, include_runinfo=include_runinfo)
        except Exception as exc:
            if view is not None:
                view.show_error(f"Prompt building failed. {type(exc).__name__}: {exc}")
            raise

        if view is not None:
            view.set_prompt(prompt)
            view.set_status("waiting for the model...")

        try:
            response = self.assistant.call_llm(prompt, include_runinfo=include_runinfo)
        except Exception as exc:
            if view is not None:
                view.show_error(f"{type(exc).__name__}: {exc}")
            raise

        if view is not None:
            view.show_result(response)
            self._live_views.append(view)

        return NotebookExtensionResult(prompt=prompt, response=response)
