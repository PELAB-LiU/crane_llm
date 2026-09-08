from __future__ import annotations

from .runinfo import collect_live_runinfo, format_runinfo_for_prompt
from .session_state import NotebookSessionState


_INTERNAL_CELL_MARKERS = (
    'from nb_extension.api import get_prompt',
    'from nb_extension.api import run_prompt',
    'from nb_extension.api import get_live_runinfo_json',
    'from nb_extension.api import reload_crane_llm',
)


def _is_internal_helper_cell(source: str) -> bool:
    stripped = source.strip()
    if not stripped:
        return True

    return any(marker in stripped for marker in _INTERNAL_CELL_MARKERS)


def build_crane_prompt(
    session_state: NotebookSessionState,
    include_runinfo: bool = True,
    shell=None,
) -> str:
    """Build a CRANE-style prompt from the current live notebook session."""

    prompt_parts = []

    prompt_parts.append("# Executed Cells:\n")
    ordered_cells = [cell for cell in session_state.ordered_executed_cells() if not _is_internal_helper_cell(cell.source)]
    if ordered_cells:
        for index, cell in enumerate(ordered_cells, start=1):
            prompt_parts.append(f"## Cell {index}:\n{cell.source}\n\n")
    else:
        prompt_parts.append("No cell has been executed\n")

    if include_runinfo:
        prompt_parts.append("# Current relevent runtime information:\n")
        target_source = session_state.target_cell.source if session_state.target_cell is not None else ""
        prompt_parts.append(format_runinfo_for_prompt(collect_live_runinfo(shell=shell, target_code=target_source)))
        prompt_parts.append("\n")

    prompt_parts.append("# Target Cell:\n")
    if session_state.target_cell is not None:
        prompt_parts.append(session_state.target_cell.source)
    else:
        prompt_parts.append("")

    return "".join(prompt_parts)
