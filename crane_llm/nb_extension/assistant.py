from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import settings as settings_module
from .llm_client import default_client, _resolve_model_name
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
        # Kept as given, not resolved, so that a model, key or endpoint set
        # after this object exists still takes effect on the next call.
        self._requested_model = model
        self.session_state = session_state or NotebookSessionState()
        # One client per mode; they differ only in their system prompt. Each is
        # stored with the settings it was built from.
        self._clients: dict = {}

    @property
    def model(self) -> str:
        return _resolve_model_name(self._requested_model)

    def build_prompt(self, shell=None, include_runinfo: bool = True) -> str:
        return build_crane_prompt(
            self.session_state, include_runinfo=include_runinfo, shell=shell
        )

    def call_llm(self, prompt: str, include_runinfo: bool = True) -> str:
        # Resolved on every call. A client built before the user set their key
        # holds no key, and reusing it would keep reporting "No API key found"
        # until the kernel restarted.
        resolved = settings_module.resolve(model=self._requested_model)

        cached = self._clients.get(include_runinfo)
        if cached is None or cached[0] != resolved:
            client = default_client(
                model=self._requested_model,
                include_runinfo=include_runinfo,
                resolved=resolved,
            )
            cached = (resolved, client)
            self._clients[include_runinfo] = cached
        return cached[1].run(prompt)

    def run(self, shell=None, include_runinfo: bool = True) -> AssistantResult:
        prompt = self.build_prompt(shell=shell, include_runinfo=include_runinfo)
        response = self.call_llm(prompt, include_runinfo=include_runinfo)
        return AssistantResult(prompt=prompt, response=response)
