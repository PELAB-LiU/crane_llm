from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from IPython import get_ipython

from .cell_filter import is_internal_helper_cell
from .session_state import NotebookSessionState


def _source_key(source: str) -> str:
    """Stable fallback id for cells the kernel did not identify.

    IPython history carries no notebook cell id, so cells replayed from history
    are keyed by their source. Keying on content also collapses repeated runs of
    an unchanged cell into a single ledger entry.
    """

    digest = hashlib.sha1(source.encode("utf-8", "replace")).hexdigest()[:16]
    return f"src-{digest}"


def _execution_succeeded(result) -> bool:
    """True when the cell ran to completion without raising.

    Cells that crashed must not enter the prompt: it presents its cell list as
    code that has 'already run successfully', which is the premise the model
    reasons from.
    """

    success = getattr(result, "success", None)
    if success is not None:
        return bool(success)

    return (
        getattr(result, "error_before_exec", None) is None
        and getattr(result, "error_in_exec", None) is None
    )


@dataclass
class HookRegistration:
    enabled: bool
    message: str = ""


class IPythonSessionTracker:
    """Track successfully executed notebook cells from a live IPython kernel."""

    def __init__(self, session_state: Optional[NotebookSessionState] = None):
        self.session_state = session_state or NotebookSessionState()
        self._registered = False
        self._kernel_session_number: Optional[int] = None
        self._shell = None
        self._callback = None

    # --- session lifecycle ---------------------------------------------

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
        """Replay this kernel session's history into the ledger.

        This is what captures cells the user ran before first invoking the
        extension. History does not record whether a cell raised, so a crashed
        cell run before the first invocation can still appear; cells executed
        afterwards go through the hook, which does check.
        """

        history_manager = getattr(shell, "history_manager", None)
        session_number = getattr(history_manager, "session_number", None)
        if history_manager is None or session_number is None:
            return

        try:
            history_rows = history_manager.get_range(session=session_number, raw=True, output=False)
        except Exception:
            return

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
            if is_internal_helper_cell(raw_cell):
                continue

            self.session_state.record_executed_cell(
                cell_id=_source_key(raw_cell),
                source=raw_cell,
                session_sequence=line_number,
                execution_count=line_number,
                cell_type="code",
            )

    # --- hook registration ---------------------------------------------

    def register(self) -> HookRegistration:
        shell = get_ipython()
        if shell is None:
            return HookRegistration(enabled=False, message="No active IPython shell")

        self._shell = shell
        self._sync_with_current_session(shell)

        if self._registered:
            return HookRegistration(enabled=True, message="Hooks already registered")

        def _post_run_cell(result):
            self._sync_with_current_session(shell)
            self._record_result(result)

        self._callback = _post_run_cell
        shell.events.register("post_run_cell", _post_run_cell)
        self._registered = True
        return HookRegistration(enabled=True, message="Hooks registered")

    def dispose(self) -> None:
        """Detach the post_run_cell hook.

        Without this, rebuilding the extension (which the reload helper does on
        every backend edit) leaves the previous hook attached. Each orphan keeps
        appending to a dead session state, so both memory and per-cell work grow
        with the number of reloads.
        """

        if not self._registered:
            return

        shell = self._shell or get_ipython()
        if shell is not None and self._callback is not None:
            try:
                shell.events.unregister("post_run_cell", self._callback)
            except Exception:
                pass

        self._registered = False
        self._callback = None

    # --- recording ------------------------------------------------------

    def _record_result(self, result) -> None:
        info = getattr(result, "info", None)
        raw_cell = getattr(info, "raw_cell", None)
        if not isinstance(raw_cell, str):
            return
        if is_internal_helper_cell(raw_cell):
            return

        source_key = _source_key(raw_cell)
        cell_id = getattr(info, "cell_id", None) or source_key

        if not _execution_succeeded(result):
            # This cell's current source did not run successfully, so any earlier
            # successful record for it is stale and must not stay in the prompt.
            self.session_state.forget_executed_cell(cell_id)
            if cell_id != source_key:
                self.session_state.forget_executed_cell(source_key)
            return

        if cell_id != source_key:
            # Merge the history-seeded record for this same source, if any.
            self.session_state.forget_executed_cell(source_key)

        self.session_state.record_executed_cell(
            cell_id=cell_id,
            source=raw_cell,
            session_sequence=self.session_state.next_session_sequence,
            execution_count=getattr(result, "execution_count", None),
            cell_type="code",
        )

    def set_target_cell(self, cell_id: str, source: str, execution_count: Optional[int] = None) -> None:
        self.session_state.set_target_cell(
            cell_id=cell_id,
            source=source,
            execution_count=execution_count,
            cell_type="code",
        )
