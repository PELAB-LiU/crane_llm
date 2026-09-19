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
        # One client per mode; they differ only in their system prompt.
        self._clients: dict = {}

    def build_prompt(self, shell=None, include_runinfo: bool = True) -> str:
        return build_crane_prompt(
            self.session_state, include_runinfo=include_runinfo, shell=shell
        )

    def call_llm(self, prompt: str, include_runinfo: bool = True) -> str:
        client = self._clients.get(include_runinfo)
        if client is None:
            client = default_openai_client(model=self.model, include_runinfo=include_runinfo)
            self._clients[include_runinfo] = client
        return client.run(prompt)

    def run(self, shell=None, include_runinfo: bool = True) -> AssistantResult:
        prompt = self.build_prompt(shell=shell, include_runinfo=include_runinfo)
        response = self.call_llm(prompt, include_runinfo=include_runinfo)
        return AssistantResult(prompt=prompt, response=response)
