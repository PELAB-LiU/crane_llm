# CRANE-LLM developer guide

This guide is for people changing or building the extension. To *use* CRANE-LLM, install the published package as described in the [user guide](../../README_PYPI.md); nothing here is needed for that.

The extension has two halves, built and reloaded in different ways. Knowing which half you are editing tells you what to run afterwards:

- a **frontend** that runs in the browser: TypeScript under `src/`, compiled with `jlpm`
- a **backend** that runs inside the notebook kernel: the Python files in this directory, never compiled

**Contents:** [1. Set up](#1-set-up) · [2. Development loop](#2-development-loop) · [3. Testing](#3-testing) · [4. Architecture](#4-architecture) · [5. Build reference](#5-build-reference) · [6. Troubleshooting](#6-troubleshooting)

## Quick reference

| What you did | What to run | What to reload |
|---|---|---|
| nothing yet | [1. Set up](#1-set-up) | start the server |
| edited `src/*.ts` | `jlpm build` | hard-refresh the browser tab |
| edited a `.py` file here | nothing | `reload_crane_llm()` in a cell |
| edited `src/ui_texts.json` | `jlpm build` | hard-refresh the browser tab, and `reload_crane_llm()` in a cell |
| edited `package.json` | `jlpm install && jlpm build` | hard-refresh the browser tab |
| pulled new commits | `jlpm install && jlpm build` | browser tab and kernel |

Run `jlpm` commands in `crane_llm/nb_extension`, and everything else from the repository root.

---

## 1. Set up

### Prerequisites

- a Python environment with **JupyterLab 4**
- **Node.js** 20.19+ or 22.12+ on `PATH`, to compile the frontend
- an **API key** for the model you want to call, set up as in the user guide's [Set up a model](../../README_PYPI.md#set-up-a-model)

Every command below must run in the environment that has JupyterLab:

```bash
conda activate crane          # whichever environment holds JupyterLab
python -c "import sys, importlib.util; print(sys.prefix, importlib.util.find_spec('jupyterlab') is not None)"
```

If that prints `False`, fix the environment before going on. On a machine with several Python installations, `pip` and `jupyter` can resolve to your environment while a bare `python` resolves to a different interpreter. Installing with the wrong `python` puts the package and the frontend bundle where JupyterLab never looks, and the extension then fails to appear with no error.

### Choose an install mode

Pick one and **do not mix them**:

- **Development install**, if you will edit any code. Your working copy stays live, and you never run `pip` again after setup.
- **Regular install**, if you cannot create symlinks or only want to run the tool from source. Every change then needs a reinstall.

A regular install puts a *copy* of the code in `site-packages`. Because a kernel puts its own working directory first on the import path, a notebook in the repository root would then import your source while a notebook anywhere else imports the copy. Mixing the two modes makes it unclear which code is running.

### Development install

```bash
python -m pip install -e ".[lab,build]"
cd crane_llm/nb_extension && jlpm install && jlpm build && cd ../..
jupyter labextension develop . --overwrite
```

The last command is required. An editable install does not install data files, so `pip install -e .` alone registers the Python backend and no frontend. `jupyter labextension develop` links the bundle into the environment instead of copying it, so later rebuilds take effect without reinstalling. You run it **once**; the link keeps pointing at your working copy. Newer versions suggest `jupyter-builder develop` instead, which does the same thing.

#### On Windows: symlink permission

`jupyter labextension develop` creates a symlink, which on Windows needs Developer Mode or an elevated shell. Without either it fails with:

```
OSError: Symlinks can be activated on Windows 10 for Python version 3.8 or higher by activating the 'Developer Mode'.
```

Do one of these, then re-run the command:

1. **Enable Developer Mode**: Settings → Privacy & security → For developers → Developer Mode. This is a one-time change.
2. **Run the command once from an Administrator PowerShell.**

If neither is possible, use the [regular install](#regular-install).

**Do not substitute a directory junction.** A junction needs no privilege, but current versions of Tornado refuse to serve files through it: on Windows `os.path.islink()` is false for a junction, so Jupyter does not allow the linked directory, and every file of the extension gets a 403. `jupyter labextension list` still shows the extension as `enabled ok`, and the only trace is a server log line such as `403 GET /lab/extensions/crane-llm-jlab/static/remoteEntry.<hash>.js ... is not in root static directory`.

`--overwrite` cannot replace an existing link: it stops with `OSError: Cannot call rmtree on a symbolic link`. To re-link, delete the old link first, which removes only the link and not its target:

```powershell
(Get-Item "$prefix\share\jupyter\labextensions\crane-llm-jlab").Delete()
```

### Regular install

Build the bundle **before** installing, because `pip` copies the built bundle into place:

```bash
cd crane_llm/nb_extension && jlpm install && jlpm build:prod && cd ../..
python -m pip install .
```

To move from a regular install to a development install:

```bash
python -m pip uninstall -y crane_llm
python -m pip install -e ".[lab,build]"
cd crane_llm/nb_extension && jlpm build && cd ../..
jupyter labextension develop . --overwrite
```

`pip uninstall` removes the bundle files it installed, which is why `jlpm build` is run again before linking.

### Confirm the install

```bash
jupyter labextension list
```

Look for `crane-llm-jlab <version> enabled ok`, under the first heading next to the `labextensions` path. A trailing `*` marks a development install.

Then start Jupyter and open a notebook. The **CRANE-LLM** button should be in the toolbar, and a CRANE-LLM panel under *View → Right Sidebar*.

---

## 2. Development loop

### Frontend changes

Frontend code is everything under `src/` and `style/`. It must be compiled and bundled:

```bash
cd crane_llm/nb_extension && jlpm build && cd ../..
```

Run `jlpm install` first if you changed `package.json` or `yarn.lock`. With a regular install, follow the build with `python -m pip install .`; adding `--no-build-isolation` cuts that from about ten seconds to under two.

Then hard-refresh the browser tab (Ctrl+Shift+R or Ctrl+F5).

### Backend changes

Backend code is every `.py` file in this directory. Nothing is compiled or reinstalled. Save the file, then in any notebook cell:

```python
from crane_llm.nb_extension.api import reload_crane_llm
reload_crane_llm()
```

This reloads the backend modules in dependency order and replaces the kernel hooks, so hooks do not accumulate.

Restart the kernel instead if you added or removed a module-level import, added a file, or changed a class that already has live instances. Restarting the kernel is always safe, just slower.

### What to reload

| Changed | Hard-refresh browser | Restart Jupyter server | Restart kernel |
|---|---|---|---|
| `src/*.ts`, `style/*.css` | yes | only if the refresh does not take | no |
| `*.py` | no | no | no, use `reload_crane_llm()` |
| `package.json` | yes | only if the refresh does not take | no |

A server restart is rarely needed. The server rescans the labextensions directory on every page request, and `jlpm build` puts a content hash in the bundle filename, so a reloaded tab always gets the new bundle; the hard refresh only avoids a cached page. If a hard refresh does not pick up the change, the bundle is probably not where JupyterLab is looking; see [Troubleshooting](#6-troubleshooting).

### Confirming which code is live

For the backend, check what the kernel imported:

```python
import crane_llm; print(crane_llm.__file__)
```

If that path is not inside your working copy, the kernel is running a copy in `site-packages` and your edits are ignored. See [Choose an install mode](#choose-an-install-mode).

For the frontend, check that JupyterLab reads the bundle you built, and that the bundle is newer than your edit:

```bash
jupyter labextension list        # the path shown should be your working copy
ls -l crane_llm/nb_extension/labextension/static/
```

If the button still behaves the old way after a hard refresh, look for errors from `crane-llm-jlab` in the browser's developer console.

---

## 3. Testing

From the repository root:

```bash
pip install -e ".[test]"
python -m pytest
cd crane_llm/nb_extension && jlpm build && cd ../..
```

`tests/test_nb_extension.py` covers prompt assembly, the executed-cell ledger, the built-in checks (each reported crash is also run for real, and so is each cell they must stay silent on), provenance and origin tracing, cells that cannot be parsed, hostile kernel namespaces, and how the API key, model and endpoint are resolved. It makes no LLM call and does not read or write your own `~/.crane_llm/config.json`.

---

## 4. Architecture

### Files

| File | Role |
|---|---|
| `src/index.ts` | toolbar button, sidebar, per-cell verdict boxes, origin marks, guard output renderer |
| `style/index.css` | the look of all of the above, with the verdict colours |
| `src/ui_texts.json` | every text users see, for both the frontend and the magic; change wording here, not in code |
| `texts.py` | reads `src/ui_texts.json` for the backend |
| `api.py` | kernel-facing entry points; the frontend contract |
| `extension.py` | coordinates tracking and the magic's output |
| `magic.py` | the `%%crane_llm` and `%crane_llm` magics |
| `assistant.py` | judges a cell: built-in checks first, then the model; locates origins |
| `checker/walker.py` | the built-in checker's engine: walks the cell and decides where to stop |
| `checker/rules/` | the checker's rules, one module per library (`python.py`, `pandas.py`, `numpy.py`, `sklearn.py`, ...) |
| `checker/sites.py` | what rules are written with: the kinds of site and the value tests |
| `checker/pure.py` | the operations the walk may continue past |
| `guard.py` | checks every cell before it runs, and stops one with a certain crash |
| `provenance.py` | records what each cell did to the namespace; traces variables back to cells |
| `verdict.py` | one verdict shape for checks and model responses |
| `ipython_hooks.py` | records executed cells and their effects from the kernel |
| `session_state.py` | one ledger entry per cell, not per execution, plus the provenance log |
| `prompt_builder.py` | assembles the CRANE prompt |
| `runinfo.py` | live-namespace runtime summary |
| `cell_filter.py` | excludes the extension's own helper cells |
| `ui.py` | the `%%crane_llm` magic's output |
| `llm_client.py` | LLM clients: the OpenAI Responses API when no `base_url` is set, as in the paper's experiments, and Chat Completions otherwise, since almost no compatible server implements `/responses` |
| `settings.py` | resolves the API key, model, endpoint and API style, in the order the user guide lists |

Runtime summarisation is shared with the offline experiment pipeline through `crane_llm/runinfo_parser/runtime_summary.py`, and retry handling through `crane_llm/llms/retry.py`, so the extension and the batch experiments cannot drift apart.

### How the frontend talks to the kernel

The frontend runs a short snippet in the user's kernel and reads a delimited JSON payload back off stdout. `api.py` is therefore a contract: renaming things there breaks the button. The frontend loads the backend into each kernel as soon as the kernel is idle, so that provenance is recorded from the start of the session rather than from the first check.

A check refuses to start unless the notebook is idle. Reading the namespace means running code in the kernel, and a kernel serves requests strictly in order, so the request would wait behind the running cells and then report the namespace as it is *afterwards*, which is not the state the user asked about. It also refuses when there is no kernel or while one is restarting.

The three switches are stored in the browser. *Check cells before they run* is also sent to each notebook's kernel when the backend loads there and whenever it changes, because the guard itself lives in the kernel and is lost when the kernel restarts (`set_guard` in `api.py`).

With runtime information off, the same cells and target cell are sent, and the system prompt switches to a variant that does not tell the model to expect runtime information.

The executed-cell ledger is keyed on the notebook's own cell id, which the kernel receives with every execute request. Re-running a cell replaces its entry rather than appending a second one, cells that raised are excluded, and the cell under analysis is never listed as already executed. Predictions are never written into the `.ipynb`.

Lower-level entry points, for use from a notebook cell:

| Function | Purpose |
|---|---|
| `get_prompt(source, cell_id=..., include_runinfo=True)` | the assembled prompt, no LLM call |
| `run_crane_llm(source, cell_id=..., include_runinfo=True)` | prompt and response |
| `get_live_runinfo_json(target_code)` | runtime summary of the namespace |
| `reload_crane_llm()` | reload the backend in a live kernel |

### The built-in checker

`checker/walker.py` walks the target cell in Python's evaluation order against the live namespace, and reports a crash only when it reaches an operation known to raise for the values it will receive. The guarantee rests on where the walk stops: at anything that could run code whose effect is unknown, which includes calls to user functions, control flow, `try` blocks, stores into objects, and operations on values the walk did not compute. It continues only past operations that change nothing, such as `print`, `df.head()`, or constructing a scikit-learn estimator. If one of those raised instead, the cell would still crash, only earlier.

A check therefore never declares a cell safe. When the walk stops, or ends without a finding, the model is asked. Any exception inside the walk counts as no finding.

When the walk stops at a statement, it resumes after it (`_walk` and `_Resume` in `checker/walker.py`) with a fresh walker that knows only what the skipped code cannot have changed. Unknown code can rebind a notebook variable only through a statement in the cell that names it, a notebook function that assigns it as a global (`STORE_GLOBAL` in its bytecode), or code that writes the namespace directly: `globals()`, `exec`, `sys.modules['__main__']`, IPython magics. If the cell or any notebook function uses one of the latter, nothing is claimed past the stop. Otherwise every variable that could be rebound, and every value that is not immutable (numbers, strings, `None`, tuples of them), becomes unknown, and the rules run on the rest as usual. Rules about the world outside the kernel (missing files, modules and module attributes) are skipped after a stop, since the skipped code may have written the file or imported the submodule. The walk does not resume past a `while` loop, which may never end, a `raise`, or the first import of a module that is not part of Python or an installed package.

`checker/pure.py` lists the operations the walk may continue past. Adding to those lists lets the rules see further into cells, but only operations that never change anything belong there.

Each library behaviour a rule depends on was confirmed against the library itself. Some are less obvious than they look, which is why the rules have exceptions:

- `df.groupby([...])` with labels that are not columns does not raise when the list is as long as the frame: pandas then treats it as the group values.
- A `MultiIndex` and the datetime-like indexes match partial keys, so a missing label is not a certain `KeyError` there.
- Only the predict family is checked for an unfitted estimator. Stateless transformers such as `Normalizer` can `transform` without being fitted.
- The feature-count rule relies on scikit-learn's own estimator checks, which require every estimator that takes 2-D input to reject the wrong number of features. Composite estimators such as `ColumnTransformer` are exempt: they select their columns from a DataFrame by name, in any order, and ignore the rest.

### Adding a rule

Rules live in `checker/rules/`, one module per library, separate from the walk. Each is a short function registered for one kind of operation, a *site*: reading `obj[key]`, assigning `obj[key] = value`, a call, an arithmetic operation, a comparison, iterating (a `for`, a comprehension, unpacking), a truth test (an `if`, `while`, `and`, `or`, `not`), a unary operator, a `del`, or `*x` / `**x` arguments. It receives the real values and calls `site.crash(...)` when the operation will raise. Rules are tried in the order they are registered, and the first that finds a crash reports it. `checker/rules/__init__.py` explains the details, with a template. In short:

```python
@rule(Call, "my-rule")
def my_rule(site: Call) -> None:
    if site.method != "something" or not is_frame(site.receiver):
        return
    if ...:  # the call will certainly raise
        site.crash("ValueError", "what Python would say", [site.receiver_root])
```

A rule that fails with an exception of its own counts as having found nothing, so a mistake in a rule can miss a crash but never invent one. Before adding a rule:

1. Confirm the library's behaviour on every version you rely on.
2. Add the case to `test_builtin_checks_are_certain` in `tests/test_nb_extension.py`, which runs it for real to confirm the reported exception is raised.
3. If the rule could plausibly fire on code that works, add that code to `test_builtin_checks_pass_working_code`, which requires the checker to stay silent on it.

Rules that read every value of the data, such as `astype(int)`, check its size against `scan_budget()` first and skip larger data. The budget is the user's scan limit (`crane_llm.set_scan_limit(n)` or `CRANE_LLM_SCAN_LIMIT`, default one million values), read once per check.

### The guard

`guard.py` runs the checker on each cell the user runs, before it runs. It is an IPython AST transformer, because IPython applies those only to code it is about to execute. `transform_cell` would not do: the checker and the provenance tracker call it themselves to parse cells, so a guard there would also fire on those calls. The transformer sees only the parsed code, so the guard takes the raw source from the `pre_run_cell` event just before, and skips executions that do not store history (frontend requests), the extension's own cells, cell magics, and cells with a `# crane: run` line.

To stop a cell, the transformer raises `CrashPrevented`, a subclass of IPython's `InputRejected`, which is IPython's own way for a transformer to refuse a cell: nothing is executed, the execution counts as failed, and the transformer stays registered. Any other exception inside the guard counts as no finding, because IPython unregisters a transformer that raises anything else. `_render_traceback_` replaces the traceback with one line, since no line of the cell ran.

The verdict is displayed once, in two forms: HTML, which every frontend can show, and the same verdict and origins as JSON under `application/vnd.crane-llm.guard+json`. The JupyterLab extension registers a renderer for that type (`GuardOutput` in `src/index.ts`), which takes precedence over the HTML and draws the button's own verdict box, so the origin entries jump to their cells and those cells are outlined. Other frontends show the HTML. Outlines appear only for outputs produced in the current browser session; a guard output saved in the notebook and reopened later is shown greyed out.

The checker refuses to run when the shell has other AST transformers, since they could change what runs; the guard is exempt, because it changes nothing it lets through. `reload_crane_llm()` keeps the guard on and re-installs it with the reloaded code.

### Provenance

`provenance.py` takes a snapshot before each cell of every name the cell mentions: the object's id and a cheap fingerprint of its shape, columns, dtypes, length or fitted state. After the cell it compares. A different id means the cell **assigned** the name, and a different fingerprint on the same object means it **modified** it. Code that mutates a name without a visible change in the fingerprint, such as `df[c] = ...`, is recorded as **possibly modified**. Cells that raise are recorded as well, since they may have changed the state before raising. Cells that ran before the backend was loaded are read from IPython's history and analysed from their code alone.

`locate_origins` follows a blamed variable back to its last assignment, together with every change after it. For a variable that does not exist, it searches the notebook's cells, which the frontend sends with each request, for the ones that would define it.

The variables to trace come from the check that found the crash, or from the `variables` field the system prompt asks the model for. When the model leaves that empty, the names the target cell uses that appear in the model's reasoning are traced instead.

---

## 5. Build reference

### Commands

| Command | Effect |
|---|---|
| `jlpm` | JupyterLab's pinned copy of Yarn, installed by the `jupyterlab` package |
| `jlpm install` | downloads dependencies into `node_modules/` from `yarn.lock`; compiles nothing |
| `jlpm build:lib` | compiles TypeScript in `src/` to JavaScript in `lib/` |
| `jlpm build` | `build:lib`, then bundles `lib/` into `labextension/` unminified, for iterating |
| `jlpm build:prod` | cleans, then rebuilds and bundles minified, for installing |
| `jupyter labextension develop . --overwrite` | run **once**: links `labextension/` into the environment so rebuilds are picked up; on Windows see [symlink permission](#on-windows-symlink-permission) |
| `jupyter labextension list` | shows which extensions the environment will load |

### Install extras

| Extra | Provides |
|---|---|
| `lab` | JupyterLab 4, for an environment that does not have it yet |
| `build` | `jupyter-builder`, needed to rebuild the frontend |
| `test` | the test dependencies |
| `experiments` | the batch pipeline in `crane_llm/llms/llm_executor.py` |

JupyterLab is not a required dependency, so that hosted kernels such as Kaggle and Colab, which only use the magic, are not given a second Jupyter stack. Notebook 7 and the ML libraries are not dependencies either; runtime summarisation handles pandas, NumPy, PyTorch, scikit-learn and TensorFlow objects when those packages are present, and skips them when they are not.

### Pinned dependencies

`package.json` pins three transitive packages under `resolutions`. Do not remove them without checking that the build still runs.

| Pin | Reason |
|---|---|
| `@rspack/core` `2.0.2` | the bundler version `@jupyter/builder` 1.x expects |
| `glob` `13.0.5` | 13.0.6 ships an empty `dist/commonjs` while its exports map points into it, so `require('glob')` throws and `@jupyter/builder` cannot start |
| `fast-uri` `3.1.3` | 3.1.8 imports `serializePathEncoding` from a module that does not export it, so constructing the `Ajv` instance inside `@jupyter/builder` throws |

`glob` and `fast-uri` reach the build through floating version ranges in packages outside this project, so these pins are what keeps a broken upstream release out. For the same reason the project cleans build output with its own `scripts/clean.mjs` rather than `rimraf`, which would pull in `glob` through another floating range.

### The post-build repair step

`build:labextension` runs `scripts/fix-build-load.mjs` afterwards. On Windows the build does not work without it.

`@jupyter/builder` finds the bundle it just wrote with `glob.sync(path.join(staticPath, 'remoteEntry.*.js'))`. On Windows `path.join` produces backslashes, which glob 9 and later treat as escape characters rather than separators, so the pattern matches nothing. The builder then writes `load: "static"`, which JupyterLab resolves to a directory instead of a module: the extension is listed and appears in the page config, yet never loads, with no error in the server log. The repair step rewrites `load` to the real `static/remoteEntry.<hash>.js`. To check it by hand, in `crane_llm/nb_extension`:

```bash
node -e "console.log(require('./labextension/package.json').jupyterlab._build.load)"
```

---

## 6. Troubleshooting

**The toolbar button or sidebar is missing.** Check that `jupyter labextension list` shows `crane-llm-jlab` under the first heading rather than under *Other labextensions*, then hard-refresh the tab. If `jupyter labextension develop` failed on Windows, see [symlink permission](#on-windows-symlink-permission).

**Backend edits have no effect.** The kernel is probably importing a copy in `site-packages`. Check with `import crane_llm; print(crane_llm.__file__)`, and see [Choose an install mode](#choose-an-install-mode).

**Frontend edits have no effect.** This is rarely a cache problem, because the bundle filename carries a content hash. More often JupyterLab is loading the extension from somewhere other than the directory `jlpm build` writes to. Check `jupyter labextension list`, and see the next entry.

**The server logs "Build recommended" and "crane-llm-jlab content changed".** The extension has been registered as a *source* extension, which happens if you run `jupyter labextension install` or `jupyter labextension link`, as many tutorials suggest. Do not use either on this project. JupyterLab then compiles `lib/` into its own application bundle and ignores `labextension/`, so `jlpm build` has no effect, and `jupyter labextension list` shows the extension under *Other labextensions (built into JupyterLab)* or *local extensions*. To repair it, stop the Jupyter server and run:

```bash
jupyter labextension uninstall crane-llm-jlab --no-build   # drop the source registration
jupyter lab clean                                          # purge the compiled app dir
cd crane_llm/nb_extension && jlpm build && cd ../..
jupyter labextension develop . --overwrite                 # register the prebuilt extension
jupyter labextension list                                  # check
```

Do not add `--static` or `--all` to `jupyter lab clean`. `<app-dir>/static` holds JupyterLab's own application assets, installed by the `jupyterlab` wheel, and without them every page fails with `JupyterLab application assets not found`. If that has happened, restore them with:

```bash
python -m pip install --force-reinstall --no-deps jupyterlab==4.5.9
```

**"The kernel did not return a CRANE-LLM payload".** The backend cannot be imported from the kernel. Confirm with `import crane_llm.nb_extension.api` in a cell.

**A build fails inside `node_modules` with `MODULE_NOT_FOUND`, or a `TypeError` from a package you never installed.** Yarn can leave an extracted package that does not match `yarn.lock`. Force a clean tree:

```bash
cd crane_llm/nb_extension
node scripts/clean.mjs node_modules .yarn/install-state.gz
jlpm install
```

Check the result before rebuilding, for example with `node -e "console.log(require('glob/package.json').version)"`.

**Authentication errors.** See [Where settings are read from](../../README_PYPI.md#where-settings-are-read-from) in the user guide. The error text lists the ways to supply a key.

**"The model hit the output token limit".** Raise `max_output_tokens` in `crane_llm/llms/config_llms.py`. Reasoning tokens count against that budget.
