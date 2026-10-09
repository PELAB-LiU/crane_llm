from __future__ import annotations

from IPython import get_ipython
from IPython.core.magic import Magics, line_cell_magic, magics_class

from . import guard as guard_module
from .api import get_extension, set_guard
from .texts import text


@magics_class
class CraneLLMMagics(Magics):
    @line_cell_magic
    def crane_llm(self, line, cell=None):
        """Predict whether the cell body would crash, without running it.

        Usage::

            %%crane_llm
            model.fit(x, y)

        The cell is analysed, not executed. Arguments are optional:
        ``--no-runinfo`` builds the prompt from the executed cells alone,
        ``--no-llm`` runs only the built-in checker and sends nothing, and
        anything else is taken as a model name for this check, for example
        ``%%crane_llm gpt-5-mini``.

        As a line magic, ``%crane_llm guard on`` checks every cell with the
        built-in checker before it runs, and does not run a cell that would
        crash. ``%crane_llm guard off`` stops that.
        """

        if cell is None:
            self._line(line)
            return

        arguments = line.split()
        include_runinfo = "--no-runinfo" not in arguments
        use_llm = "--no-llm" not in arguments
        remaining = [a for a in arguments if a not in ("--no-runinfo", "--no-llm")]
        model = remaining[0] if remaining else None

        try:
            get_extension().run_target_cell(
                source=cell,
                shell=get_ipython(),
                render=True,
                include_runinfo=include_runinfo,
                use_llm=use_llm,
                model=model,
            )
        except Exception:
            # Already shown in the cell output, in words meant for the user.
            # A traceback on top would bury the setup instructions a missing
            # key produces, and would stop a Run All at this cell.
            pass

    @staticmethod
    def _line(line: str) -> None:
        arguments = line.split()
        if arguments not in (["guard"], ["guard", "on"], ["guard", "off"]):
            print(text("guard.magic_usage"))
            return
        if len(arguments) == 2:
            set_guard(arguments[1] == "on")
        on = guard_module.is_enabled(get_ipython())
        print(text("guard.magic_on" if on else "guard.magic_off"))


def crane_llm(cell_source: str, model: str = None, include_runinfo: bool = True, use_llm: bool = True):
    """Convenience wrapper for direct Python invocation. Returns the verdict."""

    result = get_extension().run_target_cell(
        source=cell_source,
        shell=get_ipython(),
        render=True,
        include_runinfo=include_runinfo,
        use_llm=use_llm,
        model=model,
    )
    return result.verdict


def load_ipython_extension(ipython):
    ipython.register_magics(CraneLLMMagics)
    # Start tracking now rather than at the first check, so that every cell
    # run from here on is recorded with what it did to the namespace. That
    # record is what traces a crash back to the cell it comes from.
    get_extension()
