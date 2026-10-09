from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import settings as settings_module
from .checker import CheckFinding, run_checks
from .llm_client import default_client
from .prompt_builder import build_crane_prompt
from .provenance import Origin, locate_origins
from ..runinfo_parser.runtime_summary import extract_dependencies
from .session_state import NotebookSessionState
from .texts import text
from .verdict import (
    Verdict,
    names_mentioned,
    verdict_from_finding,
    verdict_from_response,
    verdict_no_finding,
)


# Progress stages reported by ``CraneNotebookAssistant.run``. The frontend
# reads them too (STAGE_* in src/index.ts), so keep the values stable.
STAGE_CHECKING = "checking"
STAGE_NO_FINDING = "no-finding"
STAGE_BUILDING = "building"
STAGE_WAITING = "waiting"


class PromptBuildingError(RuntimeError):
    """The prompt could not be assembled, so the model was never called."""


def describe_error(exc: BaseException) -> str:
    """A failed check, in words for the user.

    A prompt-building error is already worded that way; anything else is
    named by its type, since the message alone may not say what went wrong.
    """

    if isinstance(exc, PromptBuildingError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


@dataclass
class AssistantResult:
    # Empty when a built-in check answered and no model was called.
    prompt: str
    response: str
    verdict: Verdict
    origins: List[Origin] = field(default_factory=list)


class CraneNotebookAssistant:
    """Notebook-facing CRANE assistant backend.

    Answers from the built-in checks when one finds a certain crash, and asks
    the model otherwise. Nothing is persisted to disk.
    """

    def __init__(self, session_state: Optional[NotebookSessionState] = None):
        self.session_state = session_state or NotebookSessionState()
        # One client per mode; they differ only in their system prompt. Each is
        # stored with the settings it was built from.
        self._clients: dict = {}

    def check(self, shell=None) -> Optional[CheckFinding]:
        """A certain crash in the target cell, found without the model."""

        target = self.session_state.target_cell
        if target is None or shell is None:
            return None
        return run_checks(target.source, getattr(shell, "user_ns", {}) or {}, shell=shell)

    def build_prompt(self, shell=None, include_runinfo: bool = True) -> str:
        return build_crane_prompt(
            self.session_state, include_runinfo=include_runinfo, shell=shell
        )

    def call_llm(
        self,
        prompt: str,
        include_runinfo: bool = True,
        model: Optional[str] = None,
        resolved: Optional[settings_module.Settings] = None,
    ) -> str:
        # Resolved on every call, unless the caller just did. A client built
        # before the user set their key holds no key, and reusing it would keep
        # reporting "No API key found" until the kernel restarted.
        if resolved is None:
            resolved = settings_module.resolve(model=model)

        cached = self._clients.get(include_runinfo)
        if cached is None or cached[0] != resolved:
            client = default_client(include_runinfo=include_runinfo, resolved=resolved)
            cached = (resolved, client)
            self._clients[include_runinfo] = cached
        return cached[1].run(prompt)

    def read_response(self, response: str, shell=None, model: str = "") -> Verdict:
        verdict = verdict_from_response(response, model=model)
        if verdict.tone == "crash" and not verdict.variables:
            # The model named no variables. Fall back to the ones the target
            # cell uses that its reasoning mentions.
            target = self.session_state.target_cell
            used, _ = extract_dependencies(target.source if target else "", shell=shell)
            verdict.variables = names_mentioned(verdict.reasoning, sorted(used))
        return verdict

    def origins(
        self,
        verdict: Verdict,
        shell=None,
        notebook_cells: Optional[List[Dict[str, str]]] = None,
    ) -> List[Origin]:
        """The cells that gave the blamed variables their current state."""

        if verdict.tone != "crash" or not verdict.variables:
            return []
        target = self.session_state.target_cell
        return locate_origins(
            verdict.variables,
            self.session_state.provenance,
            getattr(shell, "user_ns", {}) or {},
            notebook_cells=notebook_cells,
            target_cell_id=target.cell_id if target else None,
        )

    def run(
        self,
        shell=None,
        include_runinfo: bool = True,
        notebook_cells: Optional[List[Dict[str, str]]] = None,
        progress: Optional[Callable[[str, str, str], None]] = None,
        use_llm: bool = True,
        model: Optional[str] = None,
    ) -> AssistantResult:
        """Judge the target cell.

        ``progress(stage, prompt, model)`` is called as the work advances:
        with ``STAGE_CHECKING`` before the built-in checks, ``STAGE_NO_FINDING``
        when they found nothing and the LLM is to be asked, ``STAGE_BUILDING``
        before the prompt is built, and ``STAGE_WAITING``, with the prompt and
        the model's name, once the model is about to be called. ``prompt`` and
        ``model`` are empty for the other stages. A failure to build the prompt
        is raised as ``PromptBuildingError``; a failed model call raises
        whatever the client raised.

        ``model`` overrides the configured model for this call only.

        With ``use_llm`` off only the built-in checker runs, and nothing is
        sent anywhere. The LLM settings are then not read at all, so a missing
        key or a broken configuration file cannot get in the way.
        When it finds nothing the verdict says so, which is not a claim that
        the cell is safe.
        """

        report = progress or (lambda stage, prompt, model: None)

        # The checks read the live kernel state, which is runtime information.
        # With it switched off the model must see the code alone, as in the
        # comparison the approach is built against. Without the model, that
        # comparison does not arise, and the checker is all there is.
        finding = None
        if include_runinfo or not use_llm:
            report(STAGE_CHECKING, "", "")
            finding = self.check(shell)
            if finding is None and use_llm:
                report(STAGE_NO_FINDING, "", "")

        if finding is not None:
            prompt, response = "", ""
            verdict = verdict_from_finding(finding)
        elif not use_llm:
            prompt, response = "", ""
            verdict = verdict_no_finding()
        else:
            report(STAGE_BUILDING, "", "")
            try:
                prompt = self.build_prompt(shell=shell, include_runinfo=include_runinfo)
            except Exception as exc:
                raise PromptBuildingError(
                    text("errors.prompt_failed", error=f"{type(exc).__name__}: {exc}")
                ) from exc
            resolved = settings_module.resolve(model=model)
            report(STAGE_WAITING, prompt, resolved.model)
            response = self.call_llm(prompt, include_runinfo=include_runinfo, resolved=resolved)
            verdict = self.read_response(response, shell=shell, model=resolved.model)
            verdict.checks_ran = include_runinfo
        return AssistantResult(
            prompt=prompt,
            response=response,
            verdict=verdict,
            origins=self.origins(verdict, shell=shell, notebook_cells=notebook_cells),
        )
