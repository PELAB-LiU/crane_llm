from __future__ import annotations


class CraneNotebookUI:
    """Optional ipywidgets panel for the ``%%crane_llm`` magic.

    The native JupyterLab frontend has its own sidebar and does not use this.
    ipywidgets is imported lazily so that it stays an optional dependency: the
    toolbar button must keep working in kernels that do not have it installed.
    """

    def __init__(self):
        import ipywidgets as widgets

        self._widgets = widgets
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
        self._shown = False

    def show(self):
        """Display the panel once; later calls just reuse the live widgets."""

        if self._shown:
            return

        from IPython.display import display

        display(self.panel)
        self._shown = True

    def set_status(self, text: str):
        self.status.value = f"<b>Status:</b> {text}"

    def set_prompt(self, text: str):
        self.prompt.value = text

    def set_response(self, text: str):
        self.response.value = text
