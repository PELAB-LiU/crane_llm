"""One verdict shape for both ways a cell can be judged.

A verdict comes either from a built-in check, which is certain, or from the
model, which is a prediction. Both surfaces -- the JupyterLab frontend and the
``%%crane_llm`` output -- show which, so the verdict carries its source, and
the words that say so, ready to show.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .checker import CheckFinding
from .texts import text


SOURCE_CHECK = "check"
SOURCE_MODEL = "model"


@dataclass
class Verdict:
    tone: str  # crash | safe | unknown | none (checker only, nothing found)
    label: str
    reasoning: str
    source: str  # SOURCE_CHECK | SOURCE_MODEL
    model: str = ""
    # Notebook variables whose state the verdict blames for the crash.
    variables: List[str] = field(default_factory=list)
    rule: str = ""
    exception: str = ""
    line: Optional[int] = None
    # For a model verdict: whether the built-in checks ran first and found
    # nothing. False when runtime information was switched off.
    checks_ran: bool = False

    @property
    def certain(self) -> bool:
        """A crash found by the built-in checker. Never true of "no crash"."""

        return self.source == SOURCE_CHECK and self.tone == "crash"

    @property
    def badge(self) -> str:
        """Says who gave the verdict."""

        if self.certain:
            return text("verdict.badge_check")
        if self.source == SOURCE_CHECK:
            return text("verdict.badge_check_only")
        if self.model:
            return text("verdict.badge_model", model=self.model)
        return text("verdict.badge_model_no_name")

    @property
    def note(self) -> str:
        """Says how far the verdict can be trusted."""

        if self.certain:
            return text("verdict.note_check")
        if self.source == SOURCE_CHECK:
            return text("verdict.note_check_only")
        if self.checks_ran:
            return text("verdict.note_model_after_check")
        return text("verdict.note_model_code_only")

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(certain=self.certain, badge=self.badge, note=self.note)
        return data


def verdict_from_finding(finding: CheckFinding) -> Verdict:
    return Verdict(
        tone="crash",
        label=text("verdict.label_check_crash"),
        reasoning=finding.reasoning(),
        source=SOURCE_CHECK,
        variables=list(finding.variables),
        rule=finding.rule,
        exception=finding.exception,
        line=finding.line,
    )


def verdict_no_finding() -> Verdict:
    """The checker ran alone, with the LLM switched off, and found nothing.

    Deliberately not a "safe" verdict: the checker only ever reports crashes
    it is certain of, so finding none says nothing about the cell.
    """

    return Verdict(
        tone="none",
        label=text("verdict.label_no_finding"),
        reasoning="",
        source=SOURCE_CHECK,
    )


def verdict_from_response(response: str, model: str = "") -> Verdict:
    """Read a model response.

    ``detection`` is the key the offline experiment prompts use.
    """

    parsed = _parse_response_json(response or "")
    if parsed is not None:
        raw = parsed["prediction"] if "prediction" in parsed else parsed.get("detection")
        reasoning = str(parsed.get("reasoning") or "")
        variables = _variable_names(parsed.get("variables"))
        if raw is True or raw == "true":
            return Verdict(
                "crash", text("verdict.label_model_crash"), reasoning, SOURCE_MODEL, model, variables
            )
        if raw is False or raw == "false":
            return Verdict("safe", text("verdict.label_model_safe"), reasoning, SOURCE_MODEL, model)
    return Verdict("unknown", text("verdict.label_unknown"), "", SOURCE_MODEL, model)


def _variable_names(raw: Any) -> List[str]:
    """Base names from what the model listed: ``model.coef_`` blames ``model``."""

    if not isinstance(raw, list):
        return []
    names = []
    for item in raw:
        if not isinstance(item, str):
            continue
        match = re.match(r"\s*`?([A-Za-z_][A-Za-z0-9_]*)", item)
        if match:
            names.append(match.group(1))
    return list(dict.fromkeys(names))


def names_mentioned(text: str, candidates: Iterable[str]) -> List[str]:
    """Which of ``candidates`` appear as identifiers in ``text``."""

    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or ""))
    return [name for name in candidates if name in words]


def _strip_code_fence(text: str) -> str:
    match = re.match(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", text, re.IGNORECASE)
    return match.group(1) if match else text


def _first_json_object(text: str) -> Optional[str]:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else None


def _parse_response_json(text: str) -> Optional[Dict[str, Any]]:
    for candidate in (text, _strip_code_fence(text), _first_json_object(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
