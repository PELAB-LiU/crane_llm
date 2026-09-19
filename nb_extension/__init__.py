"""CRANE-LLM notebook extension.

The backend modules are imported directly by the notebook kernel, so the
package makes the repository's sibling top-level packages importable before any
submodule runs. That keeps ``nb_extension.runinfo`` and friends importable on
their own, instead of only when some other module happens to be imported first.
"""

from pathlib import Path
import sys


def _ensure_repo_root_on_path() -> None:
    """Put the repository root on ``sys.path``.

    ``config``, ``llms`` and ``runinfo_parser`` live beside ``nb_extension``
    rather than inside it. Appended rather than prepended so the repository
    cannot shadow installed distributions with the same generic names.
    """

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.append(repo_root)


_ensure_repo_root_on_path()


def _jupyter_labextension_paths():
    return [
        {
            "src": "labextension",
            "dest": "crane-llm-jlab",
        }
    ]


def load_ipython_extension(ipython):
    """Support ``%load_ext nb_extension``.

    Imported lazily so that merely importing the package does not pull in
    ipywidgets and the OpenAI client.
    """

    from .magic import load_ipython_extension as _load_magics

    _load_magics(ipython)
