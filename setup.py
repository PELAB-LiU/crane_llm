import os

from setuptools import find_packages, setup


HERE = os.path.abspath(os.path.dirname(__file__))

LABEXT_NAME = "crane-llm-jlab"
LABEXT_SRC = os.path.join(HERE, "nb_extension", "labextension")
LABEXT_DEST = "share/jupyter/labextensions/" + LABEXT_NAME


def labextension_data_files():
    """Map the built frontend bundle into share/jupyter/labextensions.

    JupyterLab discovers prebuilt extensions by scanning that directory under
    sys.prefix, so without these entries `pip install` would install the Python
    backend and no frontend at all.

    The bundle is a build artifact and is not in version control, so it may be
    missing. In that case we emit a warning rather than failing: the Python
    package is still usable, and `jlpm build` followed by a reinstall (or
    `jupyter labextension develop . --overwrite`) provides the frontend.
    """

    if not os.path.isdir(LABEXT_SRC):
        print(
            "WARNING: {} not found. Run `jlpm install && jlpm build` in "
            "nb_extension/ before packaging, or the JupyterLab extension will "
            "not be installed.".format(os.path.relpath(LABEXT_SRC, HERE))
        )
        return []

    entries = []
    for root, _dirs, files in os.walk(LABEXT_SRC):
        if not files:
            continue
        relative = os.path.relpath(root, LABEXT_SRC)
        destination = LABEXT_DEST if relative == os.curdir else "/".join([LABEXT_DEST] + relative.split(os.sep))
        # Source paths must be relative to setup.py so sdists stay portable.
        sources = [
            os.path.relpath(os.path.join(root, name), HERE).replace(os.sep, "/")
            for name in files
        ]
        entries.append((destination, sources))

    entries.append((LABEXT_DEST, ["nb_extension/install.json"]))
    return entries


setup(
    name="crane_llm",
    version="0.1.0",
    description="CRANE-LLM: runtime-augmented crash prediction and diagnosis for ML notebooks",
    license="BSD-3-Clause",
    python_requires=">=3.9",
    # `config`, `llms`, `runinfo_parser` and `utils` sit beside `nb_extension`
    # and are imported by it, so they have to ship with the distribution.
    packages=find_packages(exclude=["target_nbs*", "results*", "tmp*", "sas*"]),
    py_modules=["config"],
    include_package_data=True,
    install_requires=[
        "ipython>=8.5",
        "jupyterlab>=4.0.0,<5",
        "nbformat>=5.7",
        "openai>=1.40",
        "python-dotenv>=1.0",
        "rich>=13.0",
    ],
    extras_require={
        # Only the %%crane_llm magic renders an ipywidgets panel; the browser
        # extension has its own sidebar and does not need this.
        "widgets": ["ipywidgets>=8.0"],
        # Needed to rebuild the frontend bundle, not to run it.
        "build": ["jupyter-builder>=1.0.0,<2"],
        # Batch experiment pipeline (llms/llm_executor.py).
        "experiments": [
            "google-genai",
            "ollama",
            "torch",
            "transformers",
        ],
    },
    package_data={
        "nb_extension": [
            "package.json",
            "tsconfig.json",
            "install.json",
            "src/*.ts",
        ],
        "runinfo_parser": ["*.json"],
    },
    data_files=labextension_data_files(),
    classifiers=[
        "Framework :: Jupyter",
        "Framework :: Jupyter :: JupyterLab",
        "Framework :: Jupyter :: JupyterLab :: 4",
        "Framework :: Jupyter :: JupyterLab :: Extensions",
        "Framework :: Jupyter :: JupyterLab :: Extensions :: Prebuilt",
    ],
)
