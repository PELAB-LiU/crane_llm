"""CRANE-LLM notebook extension helpers.

The notebook runtime imports the submodules directly. Keep package import side
effects minimal so `from nb_extension.api import ...` works reliably in kernels
with different startup paths.
"""


def _jupyter_nbextension_paths():
    return [
        {
            "section": "notebook",
            "src": "static",
            "dest": "nb_extension",
            "require": "nb_extension/main",
        }
    ]


def _jupyter_labextension_paths():
    return [
        {
            "src": "labextension",
            "dest": "crane-llm-jlab",
        }
    ]
