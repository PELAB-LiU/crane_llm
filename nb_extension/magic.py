from __future__ import annotations

from IPython.core.magic import Magics, magics_class, cell_magic, needs_local_scope
from IPython import get_ipython

from .api import get_extension


@magics_class
class CraneLLMMagics(Magics):
    def __init__(self, shell=None):
        super().__init__(shell=shell)
        self.extension = get_extension()

    @cell_magic
    @needs_local_scope
    def crane_llm(self, line, cell, local_ns=None):
        """Run CRANE-LLM against the current cell body.

        Usage:
            %%crane_llm
            x = ...
            ...
        """

        shell = get_ipython()
        result = self.extension.run_target_cell(source=cell, shell=shell, render=True)
        return result.response


def crane_llm(cell_source: str):
    """Convenience wrapper for direct Python invocation."""

    shell = get_ipython()
    result = get_extension().run_target_cell(source=cell_source, shell=shell, render=True)
    return result.response


def load_ipython_extension(ipython):
    ipython.register_magics(CraneLLMMagics)
