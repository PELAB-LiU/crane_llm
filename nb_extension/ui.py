from __future__ import annotations

from typing import Optional

import ipywidgets as widgets
from IPython.display import display


class CraneNotebookUI:
    """Simple notebook UI for prompt/status/result display.

    The prompt and status are rendered in a compact side panel, while the final
    response is emitted into the notebook output area by the caller.
    """

    def __init__(self):
        self.status = widgets.HTML(value="<b>Status:</b> idle")
        self.prompt = widgets.Textarea(
            value="",
            description="Prompt",
            layout=widgets.Layout(width="100%", height="320px"),
            disabled=True,
        )
        self.response = widgets.Textarea(
            value="",
            description="Response",
            layout=widgets.Layout(width="100%", height="240px"),
            disabled=True,
        )
        self.panel = widgets.VBox(
            [
                widgets.HTML("<h4 style='margin:0 0 6px 0;'>CRANE-LLM</h4>"),
                self.status,
                self.prompt,
                self.response,
            ],
            layout=widgets.Layout(width="100%"),
        )

    def show(self):
        display(self.panel)

    def set_status(self, text: str):
        self.status.value = f"<b>Status:</b> {text}"

    def set_prompt(self, text: str):
        self.prompt.value = text

    def set_response(self, text: str):
        self.response.value = text
