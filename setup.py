"""Install hook for the prebuilt JupyterLab frontend.

All package metadata lives in ``pyproject.toml``. What cannot be expressed
there is ``data_files``: JupyterLab discovers prebuilt extensions by scanning
``share/jupyter/labextensions`` under ``sys.prefix``, and the set of files to
put there is only known by walking the build output.
"""

import os
import sys

from setuptools import setup


HERE = os.path.abspath(os.path.dirname(__file__))

LABEXT_NAME = "crane-llm-jlab"
LABEXT_SRC = os.path.join(HERE, "crane_llm", "nb_extension", "labextension")
LABEXT_DEST = "share/jupyter/labextensions/" + LABEXT_NAME

# Commands that produce something for other people to install. A wheel or sdist
# without the frontend bundle is the worst artifact this project can publish:
# it installs cleanly, reports success, and then no button appears. An editable
# install has no such problem, because `jupyter labextension develop` links the
# bundle separately and the developer is expected to run `jlpm build` anyway.
_DISTRIBUTION_COMMANDS = {"bdist_wheel", "sdist", "bdist_egg", "build_wheel"}

_OVERRIDE_ENV_VAR = "CRANE_LLM_ALLOW_MISSING_LABEXTENSION"


def _building_a_distribution():
    return bool(_DISTRIBUTION_COMMANDS.intersection(sys.argv))


def labextension_data_files():
    """Map the built frontend bundle into ``share/jupyter/labextensions``."""

    if not os.path.isdir(LABEXT_SRC):
        message = (
            "{} not found. Run `jlpm install && jlpm build:prod` in "
            "crane_llm/nb_extension/ before packaging, or the JupyterLab "
            "extension will not be installed.".format(
                os.path.relpath(LABEXT_SRC, HERE)
            )
        )
        if _building_a_distribution() and not os.environ.get(_OVERRIDE_ENV_VAR):
            raise SystemExit(
                "ERROR: " + message + "\n\nSet {}=1 to build a backend-only "
                "distribution anyway.".format(_OVERRIDE_ENV_VAR)
            )
        print("WARNING: " + message)
        return []

    # Written by the bundler for debugging. It is large, it changes on every
    # build, and nothing reads it at run time.
    skip = {"build_log.json"}

    entries = []
    for root, _dirs, files in os.walk(LABEXT_SRC):
        files = [name for name in files if name not in skip]
        if not files:
            continue
        relative = os.path.relpath(root, LABEXT_SRC)
        destination = (
            LABEXT_DEST
            if relative == os.curdir
            else "/".join([LABEXT_DEST] + relative.split(os.sep))
        )
        # Source paths must be relative to setup.py so sdists stay portable.
        sources = [
            os.path.relpath(os.path.join(root, name), HERE).replace(os.sep, "/")
            for name in files
        ]
        entries.append((destination, sources))

    entries.append((LABEXT_DEST, ["crane_llm/nb_extension/install.json"]))
    return entries


setup(data_files=labextension_data_files())
