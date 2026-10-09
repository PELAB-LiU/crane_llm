# CRANE-LLM Notebook Extension

A JupyterLab 4 / Notebook 7 extension that predicts whether the selected code cell will crash, using the already-executed cells and the live kernel namespace as context.

**This document is for people changing the extension.** If you only want to *use* it, install the published wheel instead of building anything: see [Installing](../../README_PYPI.md#install) in the user guide. Nothing below is needed for that.

The extension has two halves, built and reloaded in different ways. Knowing which half you are editing tells you what to run afterwards:

- a **frontend** bundle that runs in the browser, written in TypeScript under `src/`, compiled with `jlpm`
- a **backend** that runs inside the notebook kernel, written in Python in this directory, never compiled

## Quick reference

| What you did | What to run | What to reload |
|---|---|---|
| nothing yet, first setup | [section 1](#1-build-from-scratch) | start the server |
| edited `src/*.ts` | `jlpm build` | hard-refresh the browser tab |
| edited a `.py` file here | nothing | `reload_crane_llm()` in a cell |
| edited `src/ui_texts.json` | `jlpm build` | hard-refresh the browser tab, and `reload_crane_llm()` in a cell |
| edited `package.json` | `jlpm install && jlpm build` | hard-refresh the browser tab |
| pulled new commits | `jlpm install && jlpm build` | browser tab and kernel |

---

# 1. Build from scratch

Before you start you need three things:

- a Python environment with **JupyterLab 4** installed
- **Node.js** 20.19+ or 22.12+ on `PATH`, to compile the frontend
- an **API key** for whichever model you want to call; see
  [Setting up a model](../../README_PYPI.md#set-up-a-model)

## 1.0 Activate the right environment first

Every command below must run in the environment that has JupyterLab installed.

```bash
conda activate crane          # whichever environment holds JupyterLab
```

This matters more than it looks. On a machine with several Python installations, `pip` and `jupyter` often resolve to your environment while a bare `python` resolves to a different interpreter. Installing with the wrong `python` puts the package and the frontend bundle under a prefix JupyterLab never scans, and the extension then fails to appear with no error explaining why. Check before you start:

```bash
python -c "import sys, importlib.util; print(sys.prefix, importlib.util.find_spec('jupyterlab') is not None)"
```

If that prints `False`, stop and fix your environment before going on.

Two things the install deliberately does not pull in:

- **JupyterLab itself.** It is not a dependency, so that hosted kernels such as Kaggle and Colab, which only use the magic, are not handed a second Jupyter stack. The environment checked above already has it; in a fresh one, install `crane-llm[lab]`.
- **Notebook 7.** Add `pip install notebook` as well if you want the Notebook 7 interface rather than JupyterLab.
- **The ML stack.** Runtime summarisation covers pandas, numpy, torch, sklearn and TensorFlow objects when those packages are present, and quietly skips them when they are not. Your notebooks will normally have already brought whichever ones they use.

## 1.1 Choose a mode

Pick one and **do not mix them**. Appendix B explains the difference.

- **Development** if you will edit any code. Your working copy stays live, and you never run `pip` again after setup.
- **Regular** if you only want to run the tool. Every change then needs a reinstall.

## 1.2 Development setup (recommended)

```bash
conda activate crane
python -m pip install -e ".[lab,build]"
cd crane_llm/nb_extension && jlpm install && jlpm build && cd ../..
jupyter labextension develop . --overwrite
```

The last command is required, not optional. An editable install does not install data files, so `pip install -e .` on its own registers the Python backend and no frontend at all. `jupyter labextension develop` symlinks the bundle instead of copying it, which is what lets later rebuilds take effect without reinstalling.

It is also a **one-time** command. The link it creates keeps pointing at your working copy, so every later `jlpm build` is picked up through it.

## 1.2.1 If the last command fails on Windows

`jupyter labextension develop` can only do its job if it is allowed to create a symlink, which on Windows needs Developer Mode or an elevated shell. Without either you get:

```
OSError: Symlinks can be activated on Windows 10 for Python version 3.8 or higher by activating the 'Developer Mode'.
```

It has to be a **real symlink**. Pick whichever of these you can do, then re-run the `develop` command:

1. **Enable Developer Mode.** Settings, Privacy & security, For developers, Developer Mode. This is the one-time fix and needs no elevation afterwards.
2. **Run it once from an Administrator PowerShell.** Administrators hold the privilege that creating a symlink requires.

If your machine allows neither, use the regular install of
[section 1.3](#13-regular-setup) instead and reinstall after each frontend build. That copies the bundle rather than linking it, so no privilege is involved.

Two more things to know about this command:

- **`--overwrite` cannot replace an existing link.** It tries to `rmtree` the old path and stops with `OSError: Cannot call rmtree on a symbolic link`, leaving the old link in place. Delete the link first, which removes only the link and never its target:

  ```powershell
  (Get-Item "$prefix\share\jupyter\labextensions\crane-llm-jlab").Delete()
  ```

- **Newer versions print a deprecation notice.** `jupyter labextension develop` now suggests `jupyter-builder develop` instead. Both do the same thing here; the older spelling still works.

### Do not substitute a directory junction

A junction looks like the obvious workaround, since it links a local directory with no special privilege. It works only on older stacks and then fails silently, so it is worse than not linking at all.

Tornado 6.5.9 stopped serving files through a directory link unless the handler opts in. `jupyter_server` added that opt-in and `jupyterlab_server` uses it for the extensions route, but the check is `os.path.islink()`, and on Windows a junction reports `islink` as **False**. The allowed directory is therefore never widened, Tornado resolves the real path, finds it outside the extensions root, and answers 403 for every file.

The failure is hard to recognise: `jupyter labextension list` shows the extension as `enabled ok`, it appears in the page config the browser receives, and nothing appears in the UI. The only trace is one line in the server log:

```
403 GET /lab/extensions/crane-llm-jlab/static/remoteEntry.<hash>.js
  ... is not in root static directory
```

A junction still works on Tornado older than 6.5.9, which is why it can seem fine and then break on an unrelated upgrade. Use a real symlink or a copy.

## 1.3 Regular setup

Build the bundle **before** installing. `pip` copies the built bundle into place and cannot do so if it does not exist yet.

```bash
conda activate crane
cd crane_llm/nb_extension && jlpm install && jlpm build:prod && cd ../..
python -m pip install .
```

## 1.4 Confirm the install

```bash
jupyter labextension list
```

You want a line reading `crane-llm-jlab <version> enabled ok`. A trailing `*` means it is a local (development) install, which is correct for section 1.2.

Then start Jupyter and open a notebook. The **CRANE-LLM** button should be in the toolbar, and a CRANE-LLM panel available under *View, Right Sidebar*.

---

# 2. After changing frontend code

Frontend code is everything under `src/`. It must be compiled and bundled.

```bash
cd crane_llm/nb_extension
jlpm build
cd ../..
```

In the regular install of section 1.3, follow that with `python -m pip install .` as well, because the bundle was copied rather than linked. Adding `--no-build-isolation` cuts that from about ten seconds to under two.

Then reload, as described in [section 4](#4-seeing-your-changes-take-effect).

If you changed `package.json` or `yarn.lock`, run `jlpm install` first. Editing TypeScript alone does not need it.

---

# 3. After changing backend code

Backend code is every `.py` file in this directory. Nothing is compiled and nothing is reinstalled. Save the file, then in any notebook cell:

```python
from crane_llm.nb_extension.api import reload_crane_llm
reload_crane_llm()
```

This reloads the backend modules in dependency order and detaches the previous kernel hook before installing a new one, so hooks do not accumulate.

Restart the kernel instead if you added or removed a module-level import, added a new file, or changed a class that already has live instances. Restarting the kernel is always safe and is never wrong, just slower.

---

# 4. Seeing your changes take effect

Different changes need different things reloaded. If a change seems to have had no effect, this table is the first thing to check: usually the browser tab is still showing the page it loaded before the rebuild.

| Changed | Hard-refresh browser | Restart Jupyter server | Restart kernel |
|---|---|---|---|
| `src/*.ts` | yes | only if the refresh does not take | no |
| `*.py` | no | no | no, use `reload_crane_llm()` |
| `package.json` | yes | only if the refresh does not take | no |

**Hard-refresh** means Ctrl+Shift+R, or Ctrl+F5.

A server restart is usually unnecessary. The Jupyter server rescans the labextensions directory on every page request, so a reloaded tab already sees a freshly built bundle. Because `jlpm build` puts a content hash in the bundle filename, the browser cannot serve you a stale bundle either; the hard refresh is only to avoid a cached page. Restart the server if a hard refresh genuinely does not pick the change up, which usually means the bundle is not where JupyterLab is looking rather than a caching problem.

## Confirming which code is actually live

For the backend, check what the kernel imported:

```python
import crane_llm; print(crane_llm.__file__)
```

If that path is not inside your working copy, you are running a copy in `site-packages` and your edits are being ignored. See appendix B.

For the frontend, confirm the bundle on disk is newer than your edit, and that JupyterLab is reading the one you built:

```bash
jupyter labextension list        # the path shown should be your working copy
ls -l crane_llm/nb_extension/labextension/static/
```

If the button still behaves the old way after a hard refresh, open the browser's developer console and look for errors from `crane-llm-jlab`.

---

# 5. Using the extension

1. Run some code cells as usual, and let them finish.
2. Select the code cell you want to check.
3. Click **CRANE-LLM** in the toolbar, or run *Run CRANE-LLM* from the command palette.

**The notebook must be idle.** If any cell is queued or running, CRANE-LLM refuses to start and tells you so, rather than doing anything. This is not a limitation that can be worked around: reading the kernel namespace means running code in the kernel, a kernel serves requests strictly in order, so the request would wait behind your cells and then report the namespace as it is *afterwards*, which is not the state you asked about. It also refuses if there is no kernel, or while one is restarting.

The verdict appears under the selected cell, and the right sidebar shows the full prompt and the raw response. The verdict is stated in words with its reasoning, and shown as a colour both down the left edge of the cell and on the verdict box.

| Colour | Meaning |
|---|---|
| red | a crash is certain or predicted |
| green | no crash is predicted |
| amber | the response could not be read as a verdict |
| grey | the prediction has gone stale |

Amber means the model returned something the extension could not parse. The raw text is still shown in the sidebar, so you can see what came back.

A badge beside the verdict says who gave it. **Built-in check · certain** means a check in `checks.py` found the crash from the live kernel state and no model was called; the sidebar then shows no prompt. **LLM prediction · *model*** means the model was asked.

When a crash is found or predicted, the box lists the **origin of the crash**: the cells that gave the blamed variables their current state. Each entry jumps to its cell, and those cells are outlined in violet with a note until the verdict goes stale.

A prediction goes stale when you edit the cell or run any other cell, because both change what the prediction was based on. Re-running the analysed cell removes its prediction, and restarting the kernel clears all of them.

## Runtime information switch

The switch is reachable three ways, all showing the same value:

- **Hover the CRANE-LLM toolbar button.** A small panel drops down with the checkbox. Tabbing to the button opens it too, and Escape closes it.
- **The sidebar**, which has the same checkbox.
- **The command palette**, *CRANE-LLM: Include Runtime Information*, which you can bind to a key.

It is on by default and remembered across reloads.

| Switch | What goes into the prompt |
|---|---|
| on (default) | executed cells, runtime state of the names the target cell uses, target cell |
| off | executed cells and target cell only |

Turning it off asks the model to predict from the code alone. The same cells and the same target cell are sent either way, and the system prompt switches to a variant that does not tell the model to expect runtime information it is not being given. The built-in checks are skipped as well, since they read the live kernel state.

Turning it off is also worth trying when a prompt is too large, since the runtime section is usually the biggest part.

The target cell is never executed, and predictions are not saved into the `.ipynb`.

## Checking cells before they run

The third switch, *Check cells before they run*, is off by default and reachable the same three ways. With it on, the kernel runs the built-in checker on every cell just before the cell runs. A cell with a certain crash is not run at all: its output shows the verdict and the origin of the crash, followed by a `CrashPrevented` error, which stops *Run All* like the crash would have. The model is never asked, and the other two switches do not apply. A line `# crane: run` anywhere in a cell lets it run regardless.

The switch is remembered in the browser, and its value is sent to each notebook's kernel when the backend loads there and whenever it changes, since the guard itself lives in the kernel and is lost when it restarts. Where the frontend is not available, `%crane_llm guard on` does the same. How it hooks into IPython is described under [The guard](#the-guard).

## From Python

```python
%load_ext crane_llm
```

```python
%%crane_llm
model.fit(x_train, y_train)
```

The cell body is analysed, not executed. Arguments are optional: `--no-runinfo` builds the prompt from the executed cells alone, `--no-llm` runs only the built-in checker, and anything else is read as a model name, so `%%crane_llm gpt-5-mini --no-runinfo` works. The output shows the verdict with its badge, the origin of the crash, and the prompt and raw response folded under it, and turns grey once another cell runs. `%load_ext crane_llm` starts provenance tracking, so load it early in the notebook. It uses only the standard display protocol, so it works where the frontend cannot load: Kaggle, Colab, VS Code and classic Notebook. The line magic `%crane_llm guard on` (or `off`) switches checking cells before they run, until the kernel restarts; `%crane_llm guard` reports whether it is on.

---

# Appendix

## A. What the commands mean

| Command | Effect |
|---|---|
| `jlpm` | JupyterLab's pinned copy of Yarn, installed by the `jupyterlab` package |
| `jlpm install` | downloads dependencies into `node_modules/` from `yarn.lock`; compiles nothing |
| `jlpm build:lib` | compiles TypeScript in `src/` to JavaScript in `lib/` |
| `jlpm build` | `build:lib`, then bundles `lib/` into `labextension/` unminified, for iterating |
| `jlpm build:prod` | cleans, then rebuilds and bundles minified, for installing |
| `jupyter labextension develop . --overwrite` | run **once**: symlinks `labextension/` into the environment so rebuilds are picked up. Needs symlink permission; on Windows see [1.2.1](#121-if-the-last-command-fails-on-windows) |
| `jupyter labextension list` | shows which extensions the environment will load |

Optional install extras:

| Extra | Provides |
|---|---|
| `lab` | JupyterLab 4, for an environment that does not have it yet |
| `build` | `jupyter-builder`, needed to rebuild the frontend |
| `experiments` | the batch pipeline in `crane_llm/llms/llm_executor.py` |

## B. Development versus regular install, and how to switch

A regular install puts a *copy* of the code in `site-packages`. Whether your edits take effect then depends on where the notebook file sits, because a kernel puts its own working directory first on the import path. A notebook in the repository root picks up your source; one in any other directory picks up the stale copy. That ambiguity is why the two modes must not be mixed.

To move from a regular install back to a development install:

```bash
conda activate crane
python -m pip uninstall -y crane_llm
python -m pip install -e ".[lab,build]"
cd crane_llm/nb_extension && jlpm build && cd ../..
jupyter labextension develop . --overwrite
```

`pip uninstall` removes the bundle files it installed, which is why `jlpm build` is repeated before re-linking. On Windows the last command needs symlink permission; see [section 1.2.1](#121-if-the-last-command-fails-on-windows).

## C. Pinned dependencies

`package.json` pins three transitive packages under `resolutions`. Do not remove them without checking that the build still runs.

| Pin | Reason |
|---|---|
| `@rspack/core` `2.0.2` | the bundler version `@jupyter/builder` 1.x expects |
| `glob` `13.0.5` | 13.0.6 ships an empty `dist/commonjs` while its exports map points into it, so `require('glob')` throws and `@jupyter/builder` cannot start |
| `fast-uri` `3.1.3` | 3.1.8 imports `serializePathEncoding` from a module that does not export it, so constructing the `Ajv` instance inside `@jupyter/builder` throws |

Both `glob` and `fast-uri` reached the build through floating version ranges in packages we do not control, so these pins are the only thing keeping a broken upstream release out.

`scripts/clean.mjs` exists for the same reason. It replaced the `rimraf` devDependency, which pulled `glob` in through a floating range.

## C2. The post-build repair step

`build:labextension` runs `scripts/fix-build-load.mjs` afterwards. This is not optional on Windows.

`@jupyter/builder` finds the bundle it just wrote with `glob.sync(path.join(staticPath, 'remoteEntry.*.js'))`. On Windows `path.join` produces backslashes, and glob 9 and later treat a backslash as an escape character rather than a separator, so the pattern matches nothing. The builder then records an empty file name and writes `load: path.join('static', '')`, which is the bare string `"static"`.

JupyterLab resolves that to a directory instead of a module. The extension is listed by `jupyter labextension list` and appears in the page config, yet never loads in the browser, with no error in the server log. The repair step rewrites `load` to the real `static/remoteEntry.<hash>.js`. You can check it by hand:

```bash
node -e "console.log(require('./labextension/package.json').jupyterlab._build.load)"
```

## D. Architecture

| File | Role |
|---|---|
| `src/index.ts` | toolbar button, sidebar, per-cell verdict boxes, origin marks |
| `style/index.css` | the look of all of the above, with the verdict colours |
| `src/ui_texts.json` | every text users see, for both the frontend and the magic; change wording here, not in code |
| `texts.py` | reads `src/ui_texts.json` for the backend |
| `api.py` | kernel-facing entry points; the frontend contract |
| `extension.py` | coordinates tracking and the magic's output |
| `assistant.py` | judges a cell: built-in checks first, then the model; locates origins |
| `checker/walker.py` | the built-in checker's engine: walks the cell and decides where to stop |
| `checker/rules/` | the checker's rules, one module per library (`python.py`, `pandas.py`, `numpy.py`, `sklearn.py`, ...); add and edit rules here |
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
| `settings.py` | resolves the API key, model, endpoint and API style |

The frontend talks to the backend by running a short snippet in the user's kernel and reading a delimited JSON payload back off stdout. `api.py` is therefore a contract: renaming things there breaks the button. The frontend also loads the backend into each kernel as soon as it is idle, so that provenance is recorded from the start of the session rather than from the first check.

### Built-in checks

`checker/walker.py` walks the target cell in Python's evaluation order against the live namespace, and reports a crash only when it reaches an operation known to raise for the values it will receive. The guarantee rests on where the walk stops: at anything that could run code whose effect is unknown, which includes calls to user functions, control flow, `try` blocks, stores into objects, and operations on values the walk did not compute. It continues only past operations that change nothing, such as `print`, `df.head()`, or constructing a scikit-learn estimator. If one of those raised instead, the cell would still crash, only earlier.

A check therefore never declares a cell safe. When the walk stops, or ends without a finding, the model is asked as before. Any exception inside the walk counts as no finding.

When the walk stops at a statement, it resumes after it (`_walk` and `_Resume` in `checker/walker.py`), with a fresh walker that knows only what the skipped code cannot have changed. Unknown code can rebind a notebook variable only through a statement in the cell that names it, a notebook function that assigns it as a global (`STORE_GLOBAL` in its bytecode), or code that writes the namespace directly: `globals()`, `exec`, `sys.modules['__main__']`, IPython magics. If the cell or any notebook function uses one of the latter, nothing is claimed past the stop. Otherwise every variable that could be rebound, and every value that is not immutable (numbers, strings, `None`, tuples of them), becomes unknown, and the rules run on the rest as usual. The rules about the world outside the kernel, missing files and modules and module attributes, are skipped after a stop, since the skipped code may have written the file or imported the submodule. The walk does not resume past a `while` loop, which may never end, a `raise`, or the first import of a module that is not part of Python or an installed package.

Each library behaviour a rule depends on was confirmed against the library itself, and `test_builtin_checks_are_certain` in `tests/test_nb_extension.py` runs every case for real to confirm the reported exception is raised. Some behaviours are less obvious than they look, which is why the rules have exceptions:

- `df.groupby([...])` with labels that are not columns does not raise when the list is as long as the frame: pandas then treats it as the group values.
- A `MultiIndex` and the datetime-like indexes match partial keys, so a missing label is not a certain `KeyError` there.
- Only the predict family is checked for an unfitted estimator. Stateless transformers such as `Normalizer` can `transform` without being fitted.
- The feature-count rule relies on scikit-learn's own estimator checks, which require every estimator that takes 2-D input to reject the wrong number of features. Composite estimators such as `ColumnTransformer` are exempt: they select their columns from a DataFrame by name, in any order, and ignore the rest.

The rules themselves are in `checker/rules/`, separate from the walk, one module per library. Each is a short function registered for one kind of operation, a *site*: reading `obj[key]`, assigning `obj[key] = value`, a call, an arithmetic operation, a comparison, iterating (a `for`, a comprehension, unpacking), a truth test (an `if`, `while`, `and`, `or`, `not`), a unary operator, a `del`, or `*x` / `**x` arguments. It receives the real values and calls `site.crash(...)` when the operation will raise. Rules are tried in the order they are registered, and the first that finds a crash reports it. `checker/rules/__init__.py` explains how to add one, with a template. In short:

```python
@rule(Call, "my-rule")
def my_rule(site: Call) -> None:
    if site.method != "something" or not is_frame(site.receiver):
        return
    if ...:  # the call will certainly raise
        site.crash("ValueError", "what Python would say", [site.receiver_root])
```

A rule that fails with an exception of its own counts as having found nothing, so a mistake in a rule can miss a crash but never invent one. Confirm the library's behaviour on every version you rely on, and add the case to `test_builtin_checks_are_certain` in `tests/test_nb_extension.py`, which runs it for real. If the rule could plausibly fire on code that works, add that code to `test_builtin_checks_pass_working_code`, which requires the checker to stay silent on it.

Rules that read every value of the data, such as `astype(int)`, check its size against `scan_budget()` first and skip larger data. The budget is the user's scan limit: `crane_llm.set_scan_limit(n)`, or `CRANE_LLM_SCAN_LIMIT`, default one million values. It is read once per check.

`checker/pure.py` lists the operations the walk may continue past, such as `df.head()`. Adding to those lists lets the rules see further into cells, but only operations that never change anything belong there.

### The guard

`guard.py` runs the checker on each cell the user runs, before it runs. It is an IPython AST transformer, because IPython applies those only to code it is about to execute. `transform_cell` would not do: the checker and the provenance tracker call it themselves to parse cells, so a guard there would also fire on those calls, and from inside the checker. The transformer sees only the parsed code, so the guard takes the raw source from the `pre_run_cell` event just before, and skips executions that do not store history (frontend requests), the extension's own cells, cell magics, and cells with a `# crane: run` line.

To stop a cell, the transformer raises `CrashPrevented`, a subclass of IPython's `InputRejected`, which is IPython's own way for a transformer to refuse a cell: nothing is executed, the execution counts as failed, and the transformer stays registered. Any other exception inside the guard counts as no finding, because IPython unregisters a transformer that raises anything else. `_render_traceback_` replaces the traceback with one line, since no line of the cell ran.

The verdict is displayed once, in two forms: HTML, which every frontend can show, and the same verdict and origins as JSON under `application/vnd.crane-llm.guard+json`. The JupyterLab extension registers a renderer for that type in each notebook (`GuardOutput` in `src/index.ts`), which takes precedence over the HTML and draws the button's own verdict box, so the origin entries jump to their cells and those cells are outlined. Other frontends do not know the type and show the HTML. The outlines appear only for outputs produced in the current browser session, and go when another cell runs; a guard output saved in the notebook and reopened later is shown greyed out.

The checker refuses to run when the shell has AST transformers, since they could change what runs; the guard is exempt, because it changes nothing it lets through. `set_guard` in `api.py` is what the frontend calls; `reload_crane_llm()` keeps the guard on and re-installs it with the reloaded code.

### Provenance

`provenance.py` takes a snapshot before each cell of every name the cell mentions: the object's id and a cheap fingerprint of its shape, columns, dtypes, length or fitted state. After the cell it compares. A different id means the cell **assigned** the name, and a different fingerprint on the same object means it **modified** it. Code that mutates a name without a visible change in the fingerprint, such as `df[c] = ...`, is recorded as **possibly modified**. Cells that raise are recorded as well, since they may have changed the state before raising. Cells that ran before the backend was loaded are read from IPython's history and analysed from their code alone.

`locate_origins` follows a blamed variable back to its last assignment, together with every change after it. For a variable that does not exist, it searches the notebook's cells, which the frontend sends with each request, for the ones that would define it.

The variables to trace come from the check that found the crash, or from the `variables` field the system prompt asks the model for. When the model leaves that empty, the names the target cell uses that appear in the model's reasoning are traced instead.

Runtime summarisation is shared with the offline pipeline through `crane_llm/runinfo_parser/runtime_summary.py`, and retry handling through `crane_llm/llms/retry.py`, so the extension and the batch experiments cannot drift apart.

The executed-cell ledger is keyed on the notebook's own cell id, which the kernel receives with every execute request. Re-running a cell replaces its entry rather than appending a second one, cells that raised are excluded, and the cell under analysis is never listed as already executed.

Lower-level entry points, for use from a notebook cell:

| Function | Purpose |
|---|---|
| `get_prompt(source, cell_id=..., include_runinfo=True)` | the assembled prompt, no LLM call |
| `run_crane_llm(source, cell_id=..., include_runinfo=True)` | prompt and response |
| `get_live_runinfo_json(target_code)` | runtime summary of the namespace |
| `reload_crane_llm()` | reload the backend in a live kernel |

## E. Validating a change

From the repository root:

```bash
pip install -e ".[test]"
python -m pytest
cd crane_llm/nb_extension && jlpm build && cd ../..
```

`tests/test_nb_extension.py` covers prompt assembly, the executed-cell ledger, the built-in checks (each reported crash is also run for real, and so is each cell they must not report), provenance and origin tracing, cells that cannot be parsed, hostile kernel namespaces, and how the API key, model and endpoint are resolved. It makes no LLM call and does not read or write your own `~/.crane_llm/config.json`.

## F. Troubleshooting

**The toolbar button or sidebar is missing.** Check `jupyter labextension list` for `crane-llm-jlab`, and that it appears under the first heading rather than under *Other labextensions*. Then hard-refresh the tab.

**Backend edits have no effect.** You are probably running a regular install, so the kernel is importing the copy in `site-packages`. Check with `import crane_llm; print(crane_llm.__file__)` and see appendix B.

**Frontend edits have no effect.** Rarely a cache problem, because the bundle filename carries a content hash. Far more often JupyterLab is loading the extension from somewhere other than the directory `jlpm build` writes to. Check `jupyter labextension list` and see the "Build recommended" entry below.

**The server logs "Build recommended" and "crane-llm-jlab content changed".** The extension has been registered as a *source* extension, the way JupyterLab 3 did it. JupyterLab then compiles `lib/` into its own application bundle and ignores `labextension/` entirely, so `jlpm build` cannot take effect.

You get into this state by running `jupyter labextension install` or `jupyter labextension link`, which many older tutorials still recommend. Do not run either on this project; `jupyter labextension develop`, from [section 1.2](#12-development-setup-recommended), is the current equivalent.

Confirm it by looking at where `jupyter labextension list` puts the extension. Under the first heading, next to the `labextensions` path, is correct. Under *Other labextensions (built into JupyterLab)* or *local extensions* is the legacy registration.

To repair it, stop the Jupyter server and run:

```bash
jupyter labextension uninstall crane-llm-jlab --no-build   # drop the legacy entry
jupyter lab clean                                          # purge the compiled app dir
cd crane_llm/nb_extension && jlpm build && cd ../..
jupyter labextension develop . --overwrite                 # register it the modern way
jupyter labextension list                                  # check
```

This is a one-time repair. Afterwards the everyday loop is just `jlpm build` again, as in [section 2](#2-after-changing-frontend-code).

**Do not add `--static` or `--all` to `jupyter lab clean`.** Despite the name, `<app-dir>/static` is not a user build: the `jupyterlab` wheel installs it there, 295 files of it. Deleting it leaves the server serving `JupyterLab application assets not found` on every page. Restore them with:

```bash
python -m pip install --force-reinstall --no-deps jupyterlab==4.5.9
```

Plain `jupyter lab clean` is safe: it removes only the staging directory and any source extensions compiled into the app directory, and prebuilt extensions live outside it.

**`jupyter labextension develop` fails with `OSError: Symlinks can be activated on Windows 10 ... 'Developer Mode'`.** Grant the privilege and re-run it, or use a regular install; see
[section 1.2.1](#121-if-the-last-command-fails-on-windows). Do not substitute a directory junction, which is served as 403 by current versions.

**"The kernel did not return a CRANE-LLM payload".** The backend is not importable from the kernel. Confirm with `import crane_llm.nb_extension.api` in a cell.

**A build fails inside `node_modules` with `MODULE_NOT_FOUND` or a `TypeError` from a package you never installed.** Yarn can leave an extracted package that does not match `yarn.lock`, so the version on disk is not the version that was pinned. Force a clean tree:

```bash
cd crane_llm/nb_extension
node scripts/clean.mjs node_modules .yarn/install-state.gz
jlpm install
```

Check the result before rebuilding, for example `node -e "console.log(require('glob/package.json').version)"`.

**Authentication errors.** The extension looks for a key in an explicit argument, then `CRANE_LLM_API_KEY`, then `OPENAI_API_KEY`, then `~/.crane_llm/config.json`. A `.env` file is not a step of its own: it is read first and fills in whichever of those variables the environment does not already define. The error text lists the ways to supply a key. See [Setting up a model](../../README_PYPI.md#set-up-a-model).

**"The model hit the output token limit".** Raise `max_output_tokens` in `crane_llm/llms/config_llms.py`. Reasoning tokens count against that budget.
