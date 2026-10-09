"""The ``%%crane_llm`` magic's output.

The JupyterLab frontend draws its verdict and sidebar itself. Everywhere else
the extension cannot load -- Kaggle, Colab, VS Code, classic Notebook -- the
magic's cell output is all the user sees, so it has to carry the same
information: the verdict first, whether a built-in check or the model gave
it, the cells the crash comes from, and the prompt and raw response on
request.

Only the standard display protocol is used, not ipywidgets. Each verdict is
shown with a ``display_id``, which lets the kernel redraw it later, after its
cell has finished. That is how a verdict is marked stale when another cell
runs, without any frontend code.
"""

from __future__ import annotations

import html
from typing import List, Optional

from .assistant import STAGE_BUILDING, STAGE_CHECKING, STAGE_NO_FINDING, STAGE_WAITING
from .provenance import Origin
from .texts import text
from .verdict import Verdict


# The same colours as the JupyterLab frontend (style/index.css). The magic's
# output is plain HTML in frontends that load no stylesheet of ours, so they
# are written out here.
_TONE_COLORS = {
    "crash": "#dc2626",
    "safe": "#16a34a",
    "unknown": "#d97706",
    "none": "#0369a1",
    "stale": "#6b7280",
    "pending": "#6b7280",
}


_STAGE_KEYS = {
    STAGE_CHECKING: "progress.checking",
    STAGE_NO_FINDING: "progress.no_finding",
    STAGE_BUILDING: "progress.building",
}


def stage_message(stage: str, model: str) -> str:
    """One progress step, as shown here and sent to the frontend."""

    if stage == STAGE_WAITING:
        return text("progress.waiting", model=model) if model else text("progress.waiting_no_name")
    return text(_STAGE_KEYS[stage]) if stage in _STAGE_KEYS else stage


def step_label(step) -> str:
    """How an origin step is listed: its role, cell and line."""

    cell = (
        text("origins.cell_with_count", count=step.execution_count)
        if step.execution_count
        else text("origins.cell_unknown")
    )
    role = text(f"origins.roles.{step.role}")
    if step.line:
        return text("origins.step_line", role=role, cell=cell, line=step.line)
    return text("origins.step", role=role, cell=cell)


def _inline_code(sentence: str) -> str:
    """Escape ``sentence`` and set its `backticked` parts as code."""

    pieces = html.escape(sentence).split("`")
    return "".join(
        f"<code>{piece}</code>" if index % 2 else piece for index, piece in enumerate(pieces)
    )


class VerdictView:
    """One ``%%crane_llm`` result, redrawn in place as it progresses."""

    def __init__(self, footer: str = ""):
        self._handle = None
        # A last line under the verdict, such as how to get past the guard.
        self._footer = footer
        self._prompt = ""
        self._response = ""
        self._tone = "pending"
        self._label = ""
        self._body = ""
        self._verdict: Optional[Verdict] = None
        self._origins: List[Origin] = []
        # Progress steps already passed, shown while the verdict is pending.
        self._steps: List[str] = []
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

    def progress(self, stage: str, model: str) -> None:
        """Show a step. Earlier steps stay listed above it until the verdict."""

        if self._label and self._label != text("progress.starting"):
            self._steps.append(self._label)
        self.set_status(stage_message(stage, model))

    def show_result(self, verdict: Verdict, origins: List[Origin], response: str = "") -> None:
        self._steps = []
        self._verdict = verdict
        self._origins = list(origins)
        self._response = response
        self._tone, self._label, self._body = verdict.tone, verdict.label, verdict.reasoning
        self._redraw()

    def show_once(self, verdict: Verdict, origins: List[Origin], data: Optional[dict] = None) -> None:
        """Show a finished verdict that is never redrawn, such as the guard's.

        ``data`` adds other representations of it, by MIME type, for frontends
        that can draw it themselves.
        """

        from IPython.display import display

        self.show_result(verdict, origins)
        display({"text/html": self._render(), **(data or {})}, raw=True)

    def show_error(self, message: str) -> None:
        self._steps = []
        self._tone = "unknown"
        self._label = text("verdict.label_error")
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
        title = text("verdict.title_stale" if self.stale else "verdict.title", label=self._label)

        parts = [
            f'<div style="border-left:4px solid {color};padding:6px 10px;'
            f'margin:4px 0;background:{color}14;">',
        ]
        for step in self._steps:
            parts.append(
                '<div style="opacity:0.7;font-size:0.9em;">'
                f'{html.escape(text("progress.done_mark"))} {html.escape(step)}</div>'
            )
        parts += [
            '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;">',
            f'<span style="font-weight:600;color:{color};">{html.escape(title)}</span>',
        ]
        if self._verdict is not None:
            parts.append(self._render_badge(self._verdict))
        parts.append("</div>")

        if self.stale:
            parts.append(
                '<div style="opacity:0.75;font-size:0.9em;">'
                f"{html.escape(text('verdict.stale_explanation'))}</div>"
            )

        if self._body:
            parts.append(
                '<div style="white-space:pre-wrap;margin-top:4px;">'
                f"{html.escape(self._body)}</div>"
            )

        if self._verdict is not None:
            parts.append(
                '<div style="opacity:0.75;font-size:0.85em;margin-top:4px;">'
                f"{html.escape(self._verdict.note)}</div>"
            )

        if self._footer:
            parts.append(
                '<div style="margin-top:6px;font-size:0.9em;">'
                f"{_inline_code(self._footer)}</div>"
            )

        if self._origins and not self.stale:
            parts.append(self._render_origins())

        if self._prompt or self._response:
            parts.append(
                '<details style="margin-top:6px;"><summary style="cursor:pointer;">'
                f"{html.escape(text('verdict.prompt_and_response'))}</summary>"
            )
            sections = (
                (text("verdict.prompt_heading"), self._prompt),
                (text("verdict.response_heading"), self._response),
            )
            for heading, body in sections:
                if body:
                    parts.append(
                        f'<div style="margin-top:6px;font-weight:600;">{html.escape(heading)}</div>'
                        '<pre style="white-space:pre-wrap;max-height:360px;overflow:auto;'
                        'margin:2px 0;">'
                        f"{html.escape(body)}</pre>"
                    )
            parts.append("</details>")

        parts.append("</div>")
        return "".join(parts)

    @staticmethod
    def _render_badge(verdict: Verdict) -> str:
        if verdict.certain:
            style = "background:#1f2937;color:#ffffff;border:1px solid #1f2937;"
        else:
            style = "background:transparent;color:inherit;border:1px solid currentColor;opacity:0.8;"
        return (
            f'<span style="{style}border-radius:999px;padding:1px 8px;font-size:0.8em;">'
            f"{html.escape(verdict.badge)}</span>"
        )

    def _render_origins(self) -> str:
        parts = [
            '<div style="margin-top:8px;font-weight:600;">'
            f"{html.escape(text('origins.heading'))}</div>",
            '<ul style="margin:2px 0 0 0;padding-left:18px;">',
        ]
        for origin in self._origins:
            parts.append(f"<li>{html.escape(origin.summary)}<ul style='padding-left:16px;'>")
            for step in origin.steps:
                note = f" ({html.escape(step.note)})" if step.note else ""
                code = (
                    f' <code style="white-space:pre-wrap;">{html.escape(step.line_text)}</code>'
                    if step.line_text
                    else ""
                )
                parts.append(f"<li>{html.escape(step_label(step))}:{code}{note}</li>")
            parts.append("</ul></li>")
        parts.append("</ul>")
        return "".join(parts)
