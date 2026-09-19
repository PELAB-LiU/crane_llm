from __future__ import annotations

from .cell_filter import is_internal_helper_cell
from .runinfo import collect_live_runinfo, format_runinfo_for_prompt
from .session_state import NotebookSessionState


def build_crane_prompt(
    session_state: NotebookSessionState,
    include_runinfo: bool = True,
    shell=None,
) -> str:
    """Build a CRANE-style prompt from the current live notebook session."""

    prompt_parts = []
    target_cell = session_state.target_cell

    prompt_parts.append("# Executed Cells:\n")
    ordered_cells = [
        cell
        for cell in session_state.ordered_executed_cells()
        if not is_internal_helper_cell(cell.source)
        # The cell under analysis must not also be listed as already executed:
        # the prompt presents that list as code that ran successfully, which is
        # exactly the question being asked about the target.
        and not (target_cell is not None and cell.cell_id == target_cell.cell_id)
    ]
    if ordered_cells:
        for index, cell in enumerate(ordered_cells, start=1):
            prompt_parts.append(f"## Cell {index}:\n{cell.source}\n\n")
    else:
        prompt_parts.append("No cell has been executed\n")

    if include_runinfo:
        prompt_parts.append("# Current relevent runtime information:\n")
        target_source = target_cell.source if target_cell is not None else ""
        prompt_parts.append(format_runinfo_for_prompt(collect_live_runinfo(shell=shell, target_code=target_source)))
        prompt_parts.append("\n")

    prompt_parts.append("# Target Cell:\n")
    if target_cell is not None:
        prompt_parts.append(target_cell.source)
    else:
        prompt_parts.append("")

    return "".join(prompt_parts)
