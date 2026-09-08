from __future__ import annotations

from nb_extension.extension import CraneNotebookExtension
from nb_extension.prompt_builder import build_crane_prompt
from nb_extension.session_state import NotebookSessionState


def main():
    session_state = NotebookSessionState()
    session_state.record_executed_cell("cell-1", "a = 1", execution_count=1)
    session_state.set_target_cell("cell-2", "print(a)", execution_count=None)

    prompt = build_crane_prompt(session_state, include_runinfo=False, shell=None)
    assert "# Executed Cells:" in prompt
    assert "# Target Cell:" in prompt
    assert "print(a)" in prompt

    extension = CraneNotebookExtension()
    extension.session_state = session_state
    extension.assistant.session_state = session_state
    extension.assistant.call_llm = lambda built_prompt: f"SMOKE_OK\n{built_prompt[:40]}"
    result = extension.run(render=False)
    assert result.response.startswith("SMOKE_OK")
    print("CRANE-LLM notebook extension smoke test passed.")


if __name__ == "__main__":
    main()
