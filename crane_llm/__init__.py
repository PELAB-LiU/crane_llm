"""CRANE-LLM: runtime-augmented crash prediction and diagnosis for ML notebooks.

Everything the installed distribution provides lives under this one top-level
name. That is deliberate. The kernel puts the notebook's own directory ahead of
``site-packages`` on ``sys.path``, so a package installed as ``config`` or
``utils`` would be shadowed by any notebook that happens to keep a file of that
name beside it, and the extension would then fail inside the user's session for
reasons that look nothing like the cause.

Subpackages:

- ``crane_llm.nb_extension`` -- the JupyterLab extension (frontend and the
  kernel-side backend it talks to)
- ``crane_llm.runinfo_parser`` -- runtime information extraction
- ``crane_llm.llms`` -- prompts, LLM clients and the batch experiment pipeline
- ``crane_llm.config`` -- summarisation constants shared by the above
"""

__version__ = "0.0.0"

__all__ = ["__version__", "load_ipython_extension", "set_api_key"]


def _jupyter_labextension_paths():
    """Tell ``jupyter labextension develop .`` where the built bundle is.

    Declared on the top-level package because that command imports the
    distribution's root module to find this hook. ``src`` is relative to this
    file; ``dest`` is the name the bundle is served under.
    """

    return [
        {
            "src": "nb_extension/labextension",
            "dest": "crane-llm-jlab",
        }
    ]


def load_ipython_extension(ipython):
    """Support ``%load_ext crane_llm``.

    Delegates to the extension package. Imported lazily so that merely
    importing ``crane_llm`` does not pull in ipywidgets or an LLM client.
    """

    from .nb_extension import load_ipython_extension as _load

    _load(ipython)


def set_api_key(*args, **kwargs):
    """Store an API key in the user configuration file.

    Re-exported here because this is the one piece of setup a user of the
    installed package has to do, and ``crane_llm.set_api_key`` is the obvious
    place to look for it.
    """

    from .nb_extension.settings import set_api_key as _set_api_key

    return _set_api_key(*args, **kwargs)
