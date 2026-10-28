import sys

import matplotlib
from IPython import get_ipython
from IPython.terminal.embed import InteractiveShellEmbed
from IPython.utils.capture import capture_output


class IPythonExecutor:
    def __init__(self):
        self.ipython = get_ipython()
        if self.ipython is None:
            # Registered as the running shell, as in a kernel; otherwise
            # display(..., display_id=True) returns None instead of a handle,
            # and progress bars that update it raise.
            self.ipython = InteractiveShellEmbed.instance()
        self.namespace = self.ipython.user_ns  # Shared namespace

        # A Jupyter kernel imports from its current directory, so a notebook
        # can import a module beside it, or one in a folder it has %cd'ed into.
        if "" not in sys.path:
            sys.path.insert(0, "")

        # Set matplotlib backend to non-interactive
        matplotlib.use('Agg')  # prevents pop-up from plt.show()

    def run_cell(self, code: str, suppress_display: bool = True):
        """Replay a notebook cell, discarding whatever it prints or displays.

        Output is suppressed with IPython's own ``capture_output``. An earlier
        version swapped ``shell.display_pub`` for a dummy module, which does not
        implement the display publisher interface: on current IPython every cell
        that displayed anything raised ``AttributeError: 'DummyMod' object has
        no attribute 'is_publishing'``. Those failures were swallowed by the
        caller, so the cell was skipped and the extracted runtime information
        came out empty.
        """

        with capture_output(stdout=suppress_display, stderr=suppress_display, display=suppress_display):
            result = self.ipython.run_cell(code, store_history=False)

        self.namespace = self.ipython.user_ns

        if result.error_in_exec:
            raise result.error_in_exec

        return result
