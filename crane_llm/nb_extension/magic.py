from __future__ import annotations

from IPython import get_ipython
from IPython.core.magic import Magics, cell_magic, magics_class

from .api import get_extension


@magics_class
class CraneLLMMagics(Magics):
    @cell_magic
    def crane_llm(self, line, cell):
        """Predict whether the cell body would crash, without running it.

        Usage::

            %%crane_llm
            model.fit(x, y)

        The cell is analysed, not executed. Arguments are optional:
        ``--no-runinfo`` builds the prompt from the executed cells alone,
        and anything else is taken as a model name, for example
        ``%%crane_llm gpt-5-mini``.
        """

        arguments = line.split()
        include_runinfo = "--no-runinfo" not in arguments
        remaining = [a for a in arguments if a != "--no-runinfo"]
        model = remaining[0] if remaining else None

        extension = get_extension(model=model)
        try:
            extension.run_target_cell(
                source=cell,
                shell=get_ipython(),
                render=True,
                include_runinfo=include_runinfo,
            )
        except Exception:
            # Already shown in the cell output, in words meant for the user.
            # A traceback on top would bury the setup instructions a missing
            # key produces, and would stop a Run All at this cell.
            pass


def crane_llm(cell_source: str, model: str = None, include_runinfo: bool = True):
    """Convenience wrapper for direct Python invocation."""

    result = get_extension(model=model).run_target_cell(
        source=cell_source,
        shell=get_ipython(),
        render=True,
        include_runinfo=include_runinfo,
    )
    return result.response


def load_ipython_extension(ipython):
    ipython.register_magics(CraneLLMMagics)
