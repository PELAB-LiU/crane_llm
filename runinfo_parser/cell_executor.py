import io
import contextlib
import warnings
import matplotlib
from IPython import get_ipython
from IPython.terminal.embed import InteractiveShellEmbed
from IPython.core.interactiveshell import DummyMod

class IPythonExecutor:
    def __init__(self):
        self.ipython = get_ipython()
        if self.ipython is None:
            self.ipython = InteractiveShellEmbed()
        self.namespace = self.ipython.user_ns  # Shared namespace

        # Set matplotlib backend to non-interactive
        matplotlib.use('Agg')  # prevents pop-up from plt.show()

    def run_cell(self, code: str, suppress_display: bool = True):
        f = io.StringIO()
        original_display_pub = self.ipython.display_pub

        try:
            if suppress_display:
                self.ipython.display_pub = DummyMod()

            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = self.ipython.run_cell(code, store_history=False)

        finally:
            self.ipython.display_pub = original_display_pub
            self.namespace.update(self.ipython.user_ns)

        if result.error_in_exec:
            raise result.error_in_exec

        return result
