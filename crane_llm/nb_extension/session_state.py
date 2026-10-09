from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .provenance import ProvenanceLog


@dataclass
class CellRecord:
    cell_id: str
    source: str
    session_sequence: int
    execution_count: Optional[int] = None


@dataclass
class NotebookSessionState:
    """In-memory notebook session state.

    Holds one record per notebook cell that has executed *successfully* in the
    current kernel session, plus the cell currently under analysis.

    Records are keyed by ``cell_id``. Re-executing a cell replaces its previous
    record rather than appending a second one, so the prompt reflects the
    notebook as it stands instead of the full execution history.

    ``provenance`` is the opposite: every execution, failed ones included, in
    order, with what each did to the namespace. It answers which cell gave a
    variable its current state.
    """

    executed_cells: List[CellRecord] = field(default_factory=list)
    target_cell: Optional[CellRecord] = None
    next_session_sequence: int = 1
    provenance: ProvenanceLog = field(default_factory=ProvenanceLog)

    def reset(self) -> None:
        """Forget everything, for a new kernel session."""

        self.executed_cells.clear()
        self.provenance.reset()
        self.target_cell = None
        self.next_session_sequence = 1

    def _allocate_sequence(self, session_sequence: Optional[int]) -> int:
        if session_sequence is None:
            session_sequence = self.next_session_sequence
        self.next_session_sequence = max(self.next_session_sequence, session_sequence + 1)
        return session_sequence

    def record_executed_cell(
        self,
        cell_id: str,
        source: str,
        session_sequence: Optional[int] = None,
        execution_count: Optional[int] = None,
    ) -> None:
        """Record (or re-record) a successful execution of ``cell_id``."""

        session_sequence = self._allocate_sequence(session_sequence)

        record = CellRecord(
            cell_id=cell_id,
            source=source,
            session_sequence=session_sequence,
            execution_count=execution_count,
        )

        for index, existing in enumerate(self.executed_cells):
            if existing.cell_id == cell_id:
                self.executed_cells[index] = record
                return

        self.executed_cells.append(record)

    def forget_executed_cell(self, cell_id: str) -> bool:
        """Drop a cell from the executed ledger. Returns True if one was removed."""

        for index, existing in enumerate(self.executed_cells):
            if existing.cell_id == cell_id:
                del self.executed_cells[index]
                return True
        return False

    def set_target_cell(self, cell_id: str, source: str) -> None:
        self.target_cell = CellRecord(
            cell_id=cell_id,
            source=source,
            session_sequence=self.next_session_sequence,
        )

    def ordered_executed_cells(self) -> List[CellRecord]:
        return sorted(self.executed_cells, key=lambda cell: cell.session_sequence)
