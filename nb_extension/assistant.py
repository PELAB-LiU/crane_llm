from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .llm_client import default_openai_client, _resolve_model_name
from .prompt_builder import build_crane_prompt
from .session_state import NotebookSessionState


@dataclass
class AssistantResult:
    prompt: str
    response: str


class CraneNotebookAssistant:
    """Notebook-facing CRANE assistant backend.

    Builds the prompt from the live kernel session and returns the LLM
    response. Nothing is persisted to disk.
    """

    def __init__(self, model: Optional[str] = None, session_state: Optional[NotebookSessionState] = None):
        self.model = _resolve_model_name(model)
        self.session_state = session_state or NotebookSessionState()
        self._client = None

    def build_prompt(self, shell=None) -> str:
        return build_crane_prompt(self.session_state, include_runinfo=True, shell=shell)

    def call_llm(self, prompt: str) -> str:
        if self._client is None:
            self._client = default_openai_client(model=self.model)
        return self._client.run(prompt)

    def run(self, shell=None) -> AssistantResult:
        prompt = self.build_prompt(shell=shell)
        response = self.call_llm(prompt)
        return AssistantResult(prompt=prompt, response=response)
