# CRANE-LLM Notebook Extension

This package contains the native browser-based Notebook 7 / JupyterLab-style web extension for CRANE-LLM.

## Install

For a fresh Windows environment, make sure you have:

- Python environment activated (`conda activate chatgpt` in your case)
- Node.js `20.19+` or `22.12+` available on `PATH`
- JupyterLab 4 / Notebook 7 in the same environment

Use this setup sequence once from the repository root:

```bash
conda activate chatgpt
cd c:\Users\yirwa29\Downloads\Dataset-Nb\Docker_kaggle_env\crane_llm
python -m pip install -e .
cd nb_extension
jlpm install
jlpm build
cd ..
# jupyter labextension develop . --overwrite
jupyter notebook
```

Note: `jlpm build` creates the frontend bundle in `nb_extension/labextension/`. This project is a prebuilt extension, so you do not need `jupyter labextension develop . --overwrite` or `jupyter lab build --dev-build=False --minimize=False`.

If frontend TypeScript/JavaScript code changes, run the following commands to update:

```bash
cd nb_extension
jlpm build
cd ..
# jupyter labextension develop . --overwrite
jupyter notebook
```

After building, restart Notebook 7 or JupyterLab if it was already open, then hard-refresh the browser tab. The extension is loaded at app startup.

If backend Python code changes, you do not need to rebuild the frontend. Save the `.py` files, then reload the backend inside the running notebook kernel:

```python
from nb_extension.api import reload_crane_llm
reload_crane_llm()
```

If the changed code affects imports, module initialization, or you still do not see the update, restart the notebook kernel and run the helper again. 

Optionally, in a notebook cell:

```python
from nb_extension.api import load_crane_llm
load_crane_llm()
```

The `load_crane_llm()` helper is optional. It is only needed if you want to call the backend helper directly from a notebook cell. The native frontend extension does not depend on you running that helper every time.

## Load in a notebook

Load in a notebook, execute some code cells, then select a target code cell and click the CRANE-LLM toolbar button (on the top-right corner) to predict if it will crash.
The CRANE-LLM response will show under the target code cell.
The sidebar (opens via: View-Right Sidebar-Show) shows the prompt to the LLM and status and the final response.

## Validate

```bash
python -m py_compile setup.py nb_extension/__init__.py nb_extension/*.py
python -m nb_extension.smoke_test
jlpm build
```

The smoke test validates prompt assembly and the non-rendering backend path without making an LLM call.

Use `jlpm build` in `nb_extension/` after changing the frontend TypeScript/JavaScript.

If the toolbar button or sidebar does not appear, restart the Notebook 7 server and hard-refresh the browser tab. The frontend bundle is loaded at startup, so an already-open notebook will not pick up the extension until it reloads.
