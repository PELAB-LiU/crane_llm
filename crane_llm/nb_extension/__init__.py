"""CRANE-LLM notebook extension.

``config``, ``llms`` and ``runinfo_parser`` are sibling subpackages of
``crane_llm``, reached through relative imports, so nothing here depends on the
repository root being on ``sys.path``.
"""


def _jupyter_labextension_paths():
    return [
        {
            "src": "labextension",
            "dest": "crane-llm-jlab",
        }
    ]


def load_ipython_extension(ipython):
    """Support ``%load_ext crane_llm``.

    Imported lazily so that merely importing the package does not pull in
    ipywidgets and the OpenAI client.
    """

    from .magic import load_ipython_extension as _load_magics

    _load_magics(ipython)
