from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from IPython import get_ipython

from .session_state import NotebookSessionState


_INTERNAL_CELL_MARKERS = (
    'from nb_extension.api import get_prompt',
    'from nb_extension.api import run_prompt',
    'from nb_extension.api import get_live_runinfo_json',
    'from nb_extension.api import reload_crane_llm',
)


def _should_ignore_cell(source: str) -> bool:
    stripped = source.strip()
    if not stripped:
        return True

    return any(marker in stripped for marker in _INTERNAL_CELL_MARKERS)


@dataclass
class HookRegistration:
    enabled: bool
    message: str = ""


class IPythonSessionTracker:
    """Track executed notebook cells from a live IPython kernel session."""

    def __init__(self, session_state: Optional[NotebookSessionState] = None):
        self.session_state = session_state or NotebookSessionState()
        self._registered = False
        self._kernel_session_number: Optional[int] = None

    def _sync_with_current_session(self, shell) -> None:
        history_manager = getattr(shell, "history_manager", None)
        session_number = getattr(history_manager, "session_number", None)

        if session_number is None:
            if self._kernel_session_number is None:
                self.session_state.reset_for_session(None)
                self._kernel_session_number = None
            return

        if self._kernel_session_number != session_number:
            self.session_state.reset_for_session(session_number)
            self._kernel_session_number = session_number
            self._seed_current_session_history(shell)

    def _seed_current_session_history(self, shell) -> None:
        history_manager = getattr(shell, "history_manager", None)
        session_number = getattr(history_manager, "session_number", None)
        if history_manager is None or session_number is None:
            return

        history_rows = history_manager.get_range(session=session_number, raw=True, output=False)

        for row in history_rows:
            if len(row) == 3:
                _, line_number, raw_cell = row
            elif len(row) >= 2:
                line_number = row[0]
                raw_cell = row[1]
            else:
                continue

            if not isinstance(raw_cell, str) or not raw_cell.strip():
                continue
            if _should_ignore_cell(raw_cell):
                continue

            self.session_state.record_executed_cell(
                cell_id=f"history-{line_number}",
                source=raw_cell,
                session_sequence=line_number,
                execution_count=line_number,
                cell_type="code",
            )

    def register(self) -> HookRegistration:
        shell = get_ipython()
        if shell is None:
            return HookRegistration(enabled=False, message="No active IPython shell")

        self._sync_with_current_session(shell)

        if self._registered:
            return HookRegistration(enabled=True, message="Hooks already registered")

        def _post_run_cell(result):
            self._sync_with_current_session(shell)
            raw_cell = getattr(getattr(result, "info", None), "raw_cell", None)
            if raw_cell is None:
                return
            if _should_ignore_cell(raw_cell):
                return

            execution_count = getattr(result, "execution_count", None)
            cell_id = f"exec-{self.session_state.next_session_sequence}"
            self.session_state.record_executed_cell(
                cell_id=cell_id,
                source=raw_cell,
                session_sequence=self.session_state.next_session_sequence,
                execution_count=execution_count,
                cell_type="code",
            )

        shell.events.register("post_run_cell", _post_run_cell)
        self._registered = True
        return HookRegistration(enabled=True, message="Hooks registered")

    def set_target_cell(self, cell_id: str, source: str, execution_count: Optional[int] = None) -> None:
        self.session_state.set_target_cell(
            cell_id=cell_id,
            source=source,
            execution_count=execution_count,
            cell_type="code",
        )
