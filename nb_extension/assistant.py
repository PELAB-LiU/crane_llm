from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional



def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from llms.config_llms import config

from .llm_client import default_openai_client, _resolve_model_name
from .prompt_builder import build_crane_prompt
from .session_state import NotebookSessionState


@dataclass
class AssistantResult:
    prompt: str
    response: str


class CraneNotebookAssistant:
    """Notebook-facing CRANE assistant backend.

    This is the first implementation slice for the notebook extension. It does
    not persist to disk; it only builds the prompt from the live session and
    returns the LLM response for the UI to display.
    """

    def __init__(self, model: Optional[str] = None, session_state: Optional[NotebookSessionState] = None):
        self.model = _resolve_model_name(model)
        self.session_state = session_state or NotebookSessionState()

    def build_prompt(self, shell=None) -> str:
        return build_crane_prompt(self.session_state, include_runinfo=True, shell=shell)

    def call_llm(self, prompt: str) -> str:
        client = default_openai_client(model=self.model)
        return client.run(prompt)

    def run(self, shell=None) -> AssistantResult:
        prompt = self.build_prompt(shell=shell)
        response = self.call_llm(prompt)
        return AssistantResult(prompt=prompt, response=response)
