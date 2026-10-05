from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from IPython import get_ipython

from .assistant import CraneNotebookAssistant, PromptBuildingError
from .cell_filter import is_internal_helper_cell
from .ipython_hooks import IPythonSessionTracker
from .provenance import Origin
from .session_state import NotebookSessionState
from .texts import text
from .verdict import Verdict


@dataclass
class NotebookExtensionResult:
    # Both empty when a built-in check answered without calling the model.
    prompt: str
    response: str
    verdict: Verdict
    origins: List[Origin] = field(default_factory=list)


class CraneNotebookExtension:
    """High-level notebook entry point.

    Coordinates session tracking, the built-in checks, prompt generation, the
    magic's cell output, and the single-shot LLM call.
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
        notebook_cells: Optional[List[Dict[str, str]]] = None,
        use_llm: bool = True,
    ) -> NotebookExtensionResult:
        self.set_target_cell(cell_id=cell_id, source=source, execution_count=execution_count)
        return self.run(
            shell=shell,
            render=render,
            include_runinfo=include_runinfo,
            notebook_cells=notebook_cells,
            use_llm=use_llm,
        )

    def run(
        self,
        shell=None,
        render: bool = True,
        include_runinfo: bool = True,
        notebook_cells: Optional[List[Dict[str, str]]] = None,
        use_llm: bool = True,
    ) -> NotebookExtensionResult:
        """Judge the target cell: built-in checks first, then the model.

        With ``render`` the progress and the verdict are shown as this cell's
        output. Failures are shown there too and then re-raised, so a caller
        can still tell that no prediction was made.
        """

        view = None
        if render:
            from .ui import VerdictView

            view = VerdictView()
            view.show(text("progress.starting"))

        def progress(stage: str, prompt: str) -> None:
            if view is None:
                return
            if prompt:
                view.set_prompt(prompt)
            view.progress(stage, self.model)

        try:
            result = self.assistant.run(
                shell=shell,
                include_runinfo=include_runinfo,
                notebook_cells=notebook_cells,
                progress=progress,
                use_llm=use_llm,
            )
        except PromptBuildingError as exc:
            if view is not None:
                view.show_error(str(exc))
            raise
        except Exception as exc:
            if view is not None:
                view.show_error(f"{type(exc).__name__}: {exc}")
            raise

        if view is not None:
            view.show_result(result.verdict, result.origins, result.response)
            self._live_views.append(view)

        return NotebookExtensionResult(
            prompt=result.prompt,
            response=result.response,
            verdict=result.verdict,
            origins=result.origins,
        )
