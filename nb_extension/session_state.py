from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CellRecord:
    cell_id: str
    source: str
    session_sequence: int
    execution_count: Optional[int] = None
    cell_type: str = "code"


@dataclass
class NotebookSessionState:
    """In-memory notebook session state.

    The frontend extension can push executed cells here in execution order,
    and the assistant can use the current cursor cell as the target cell.
    """

    executed_cells: List[CellRecord] = field(default_factory=list)
    target_cell: Optional[CellRecord] = None
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    current_session_number: Optional[int] = None
    next_session_sequence: int = 1

    def reset_for_session(self, session_number: Optional[int] = None) -> None:
        self.executed_cells.clear()
        self.target_cell = None
        self.current_session_number = session_number
        self.next_session_sequence = 1

    def record_executed_cell(
        self,
        cell_id: str,
        source: str,
        session_sequence: Optional[int] = None,
        execution_count: Optional[int] = None,
        cell_type: str = "code",
    ) -> None:
        if session_sequence is None:
            session_sequence = self.next_session_sequence
            self.next_session_sequence += 1
        else:
            self.next_session_sequence = max(self.next_session_sequence, session_sequence + 1)

        self.executed_cells.append(
            CellRecord(
                cell_id=cell_id,
                source=source,
                session_sequence=session_sequence,
                execution_count=execution_count,
                cell_type=cell_type,
            )
        )

    def set_target_cell(
        self,
        cell_id: str,
        source: str,
        execution_count: Optional[int] = None,
        cell_type: str = "code",
    ) -> None:
        self.target_cell = CellRecord(
            cell_id=cell_id,
            source=source,
            session_sequence=self.next_session_sequence,
            execution_count=execution_count,
            cell_type=cell_type,
        )

    def ordered_executed_cells(self) -> List[CellRecord]:
        return sorted(self.executed_cells, key=lambda cell: cell.session_sequence)
