"""The ``%%crane_llm`` magic's output.

The JupyterLab frontend draws its verdict and sidebar itself. Everywhere else
the extension cannot load -- Kaggle, Colab, VS Code, classic Notebook -- the
magic's cell output is all the user sees, so it has to carry the same
information: the verdict first, and the prompt and raw response on request.

Only the standard display protocol is used, not ipywidgets. Each verdict is
shown with a ``display_id``, which lets the kernel redraw it later, after its
cell has finished. That is how a verdict is marked stale when another cell
runs, without any frontend code.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any, Dict, Optional, Tuple


# The same colours as the JupyterLab frontend (getToneColor in src/index.ts).
_TONE_COLORS = {
    "crash": "#dc2626",
    "safe": "#16a34a",
    "unknown": "#d97706",
    "stale": "#6b7280",
    "pending": "#6b7280",
}


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


def read_verdict(response: str) -> Tuple[str, str, str]:
    """``(tone, label, reasoning)`` for a model response.

    Mirrors ``readVerdict`` in the frontend, so both surfaces read a response
    the same way. ``detection`` is the key the offline experiment prompts use.
    """

    parsed = _parse_response_json(response or "")
    if parsed is not None:
        raw = parsed["prediction"] if "prediction" in parsed else parsed.get("detection")
        reasoning = str(parsed.get("reasoning") or "")
        if raw is True or raw == "true":
            return "crash", "crash predicted", reasoning
        if raw is False or raw == "false":
            return "safe", "no crash predicted", reasoning
    return "unknown", "response not understood", ""


class VerdictView:
    """One ``%%crane_llm`` result, redrawn in place as it progresses."""

    def __init__(self):
        self._handle = None
        self._prompt = ""
        self._response = ""
        self._tone = "pending"
        self._label = ""
        self._body = ""
        self.stale = False

    # --- lifecycle -------------------------------------------------------

    def show(self, status: str) -> None:
        from IPython.display import HTML, display

        self._label = status
        self._handle = display(HTML(self._render()), display_id=True)

    def set_status(self, status: str) -> None:
        self._label = status
        self._redraw()

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt

    def show_result(self, response: str) -> None:
        self._response = response
        self._tone, self._label, self._body = read_verdict(response)
        self._redraw()

    def show_error(self, message: str) -> None:
        self._tone = "unknown"
        self._label = "could not get a prediction"
        self._body = message
        self._redraw()

    def mark_stale(self) -> None:
        """Grey the verdict out: the kernel state it rested on has changed."""

        if self.stale:
            return
        self.stale = True
        self._redraw()

    # --- rendering -------------------------------------------------------

    def _redraw(self) -> None:
        if self._handle is None:
            return
        from IPython.display import HTML

        try:
            self._handle.update(HTML(self._render()))
        except Exception:
            # A frontend that has dropped the output, or a kernel shutting
            # down, is not a reason to fail the user's cell.
            pass

    def _render(self) -> str:
        tone = "stale" if self.stale else self._tone
        color = _TONE_COLORS[tone]
        title = f"CRANE-LLM: {self._label}"
        if self.stale:
            title += " (stale)"

        parts = [
            f'<div style="border-left:4px solid {color};padding:6px 10px;'
            f'margin:4px 0;background:{color}14;">',
            f'<div style="font-weight:600;color:{color};">{html.escape(title)}</div>',
        ]

        if self.stale:
            parts.append(
                '<div style="opacity:0.75;font-size:0.9em;">A cell has run since this '
                "check, so the kernel state it was based on may have changed. Run the "
                "check again for a current prediction.</div>"
            )

        if self._body:
            parts.append(
                '<div style="white-space:pre-wrap;margin-top:4px;">'
                f"{html.escape(self._body)}</div>"
            )

        if self._prompt or self._response:
            parts.append(
                '<details style="margin-top:6px;"><summary style="cursor:pointer;">'
                "Prompt and raw response</summary>"
            )
            for heading, text in (("Prompt", self._prompt), ("Response", self._response)):
                if text:
                    parts.append(
                        f'<div style="margin-top:6px;font-weight:600;">{heading}</div>'
                        '<pre style="white-space:pre-wrap;max-height:360px;overflow:auto;'
                        'margin:2px 0;">'
                        f"{html.escape(text)}</pre>"
                    )
            parts.append("</details>")

        parts.append("</div>")
        return "".join(parts)
