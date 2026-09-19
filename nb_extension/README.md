# CRANE-LLM Notebook Extension

A JupyterLab 4 / Notebook 7 extension that predicts whether the selected code
cell will crash, using the already-executed cells and the live kernel namespace
as context.

It has two halves that are built and reloaded differently, which is the source
of most confusion:

- a **frontend** bundle that runs in the browser, written in TypeScript under
  `src/`, compiled with `jlpm`
- a **backend** that runs inside the notebook kernel, written in Python in this
  directory, never compiled

## Quick reference

| What you did | What to run | What to reload |
|---|---|---|
| nothing yet, first setup | [section 1](#1-build-from-scratch) | start the server |
| edited `src/*.ts` | `jlpm build` | hard-refresh the browser tab |
| edited a `.py` file here | nothing | `reload_crane_llm()` in a cell |
| edited `package.json` | `jlpm install && jlpm build` | hard-refresh the browser tab |
| pulled new commits | `jlpm install && jlpm build` | browser tab and kernel |

---

# 1. Build from scratch

## 1.0 Activate the right environment first

Every command below must run in the environment that has JupyterLab installed.

```bash
conda activate crane          # whichever environment holds JupyterLab
```

This matters more than it looks. On a machine with several Python
installations, `pip` and `jupyter` often resolve to your environment while a
bare `python` resolves to a different interpreter. Installing with the wrong
`python` puts the package and the frontend bundle under a prefix JupyterLab
never scans, and the extension then fails to appear with no error explaining
why. Check before you start:

```bash
python -c "import sys, importlib.util; print(sys.prefix, importlib.util.find_spec('jupyterlab') is not None)"
```

If that prints `False`, stop and fix your environment. You also need Node.js
20.19+ or 22.12+ on `PATH`, and an OpenAI API key in the environment or in the
repository `.env`.

Two things the install deliberately does not pull in:

- **Notebook 7.** Installing this package brings JupyterLab, which is all the
  extension needs. Add `pip install notebook` as well if you want the Notebook 7
  interface rather than JupyterLab.
- **The ML stack.** Runtime summarisation covers pandas, numpy, torch, sklearn
  and TensorFlow objects when those packages are present, and quietly skips
  them when they are not. Your notebooks will normally have already brought
  whichever ones they use.

## 1.1 Choose a mode

Pick one and **do not mix them**. Appendix B explains the difference.

- **Development** if you will edit any code. Your working copy stays live, and
  you never run `pip` again after setup.
- **Regular** if you only want to run the tool. Every change then needs a
  reinstall.

## 1.2 Development setup (recommended)

```bash
conda activate crane
python -m pip install -e ".[widgets,build]"
cd nb_extension && jlpm install && jlpm build && cd ..
jupyter labextension develop . --overwrite
```

The last command is required, not optional. An editable install does not
install data files, so `pip install -e .` on its own registers the Python
backend and no frontend at all. `jupyter labextension develop` symlinks the
bundle instead of copying it, which is what lets later rebuilds take effect
without reinstalling.

It is also a **one-time** command. The link it creates keeps pointing at your
working copy, so every later `jlpm build` is picked up through it.

## 1.2.1 If the last command fails on Windows

`jupyter labextension develop` can only do its job if it is allowed to create a
symlink, which on Windows needs Developer Mode or an elevated shell. Without
either you get:

```
OSError: Symlinks can be activated on Windows 10 for Python version 3.8 or
higher by activating the 'Developer Mode'.
```

It has to be a **real symlink**. Pick whichever of these you can do, then re-run
the `develop` command:

1. **Enable Developer Mode.** Settings, Privacy & security, For developers,
   Developer Mode. This is the one-time fix and needs no elevation afterwards.
2. **Run it once from an Administrator PowerShell.** Administrators hold the
   privilege that creating a symlink requires.

If your machine allows neither, use the regular install of
[section 1.3](#13-regular-setup) instead and reinstall after each frontend
build. That copies the bundle rather than linking it, so no privilege is
involved.

Two things that trip people up here:

- **`--overwrite` cannot replace an existing link.** It tries to `rmtree` the
  old path and stops with `OSError: Cannot call rmtree on a symbolic link`,
  leaving the old link in place. Delete the link first, which removes only the
  link and never its target:

  ```powershell
  (Get-Item "$prefix\share\jupyter\labextensions\crane-llm-jlab").Delete()
  ```

- **Newer versions print a deprecation notice.** `jupyter labextension develop`
  now suggests `jupyter-builder develop` instead. Both do the same thing here;
  the older spelling still works.

### Do not substitute a directory junction

A junction looks like the obvious workaround, since it links a local directory
with no special privilege. It works only on older stacks and then fails
silently, so it is worse than not linking at all.

Tornado 6.5.9 stopped serving files through a directory link unless the handler
opts in. `jupyter_server` added that opt-in and `jupyterlab_server` uses it for
the extensions route, but the check is `os.path.islink()`, and on Windows a
junction reports `islink` as **False**. The allowed directory is therefore never
widened, Tornado resolves the real path, finds it outside the extensions root,
and answers 403 for every file.

The symptom is nasty: `jupyter labextension list` shows the extension as
`enabled ok`, it appears in the page config the browser receives, and nothing
in the UI appears. The only trace is one line in the server log:

```
403 GET /lab/extensions/crane-llm-jlab/static/remoteEntry.<hash>.js
  ... is not in root static directory
```

A junction still works on Tornado older than 6.5.9, which is why it can seem
fine and then break on an unrelated upgrade. Use a real symlink or a copy.

## 1.3 Regular setup

Build the bundle **before** installing. `pip` copies the built bundle into
place and cannot do so if it does not exist yet.

```bash
conda activate crane
cd nb_extension && jlpm install && jlpm build:prod && cd ..
python -m pip install .
```

## 1.4 Confirm the install

```bash
jupyter labextension list
```

You want a line reading `crane-llm-jlab v0.1.0 enabled ok`. A trailing `*`
means it is a local (development) install, which is correct for section 1.2.

Then start Jupyter and open a notebook. The **CRANE-LLM** button should be in
the toolbar, and a CRANE-LLM panel available under *View, Right Sidebar*.

---

# 2. After changing frontend code

Frontend code is everything under `src/`. It must be compiled and bundled.

```bash
cd nb_extension
jlpm build
cd ..
```

In the regular install of section 1.3, follow that with `python -m pip install .`
as well, because the bundle was copied rather than linked. Adding
`--no-build-isolation` cuts that from about ten seconds to under two.

Then reload, as described in [section 4](#4-seeing-your-changes-take-effect).

If you changed `package.json` or `yarn.lock`, run `jlpm install` first.
Editing TypeScript alone does not need it.

---

# 3. After changing backend code

Backend code is every `.py` file in this directory. Nothing is compiled and
nothing is reinstalled. Save the file, then in any notebook cell:

```python
from nb_extension.api import reload_crane_llm
reload_crane_llm()
```

This reloads the backend modules in dependency order and detaches the previous
kernel hook before installing a new one, so hooks do not accumulate.

Restart the kernel instead if you added or removed a module-level import, added
a new file, or changed a class that already has live instances. Restarting the
kernel is always safe and is never wrong, just slower.

---

# 4. Seeing your changes take effect

Different changes need different things reloaded. Nothing here is optional: the
most common "my fix did nothing" is a stale bundle in the browser cache.

| Changed | Hard-refresh browser | Restart Jupyter server | Restart kernel |
|---|---|---|---|
| `src/*.ts` | yes | only if the refresh does not take | no |
| `*.py` | no | no | no, use `reload_crane_llm()` |
| `package.json` | yes | only if the refresh does not take | no |

**Hard-refresh** means Ctrl+Shift+R, or Ctrl+F5.

A server restart is usually unnecessary. The Jupyter server rescans the
labextensions directory on every page request, so a reloaded tab already sees
a freshly built bundle. Because `jlpm build` puts a content hash in the bundle
filename, the browser cannot serve you a stale bundle either; the hard refresh
is only to avoid a cached page. Restart the server if a hard refresh genuinely
does not pick the change up, which usually means the bundle is not where
JupyterLab is looking rather than a caching problem.

## Confirming which code is actually live

For the backend, check what the kernel imported:

```python
import nb_extension; print(nb_extension.__file__)
```

If that path is not inside your working copy, you are running a copy in
`site-packages` and your edits are being ignored. See appendix B.

For the frontend, confirm the bundle on disk is newer than your edit, and that
JupyterLab is reading the one you built:

```bash
jupyter labextension list        # the path shown should be your working copy
ls -l nb_extension/labextension/static/
```

If the button still behaves the old way after a hard refresh, open the
browser's developer console and look for errors from `crane-llm-jlab`.

---

# 5. Using the extension

1. Run some code cells as usual, and let them finish.
2. Select the code cell you want to check.
3. Click **CRANE-LLM** in the toolbar, or run *Run CRANE-LLM* from the command
   palette.

**The notebook must be idle.** If any cell is queued or running, CRANE-LLM
refuses to start and tells you so, rather than doing anything. This is not a
limitation that can be worked around: reading the kernel namespace means
running code in the kernel, a kernel serves requests strictly in order, so the
request would wait behind your cells and then report the namespace as it is
*afterwards*, which is not the state you asked about. It also refuses if there
is no kernel, or while one is restarting.

The prediction appears under the selected cell, and the right sidebar shows the
full prompt and the raw response. The verdict is stated in words above the
response, and shown as a colour both down the left edge of the cell and on the
response box.

| Colour | Meaning |
|---|---|
| red | a crash is predicted |
| green | no crash is predicted |
| amber | the response could not be read as a verdict |
| grey | the prediction has gone stale |

Amber means the model returned something the extension could not parse. The raw
text is still shown, so you can see what came back.

A prediction goes stale when you edit the cell or run any other cell, because
both change what the prediction was based on. Re-running the analysed cell
removes its prediction, and restarting the kernel clears all of them.

## Runtime information switch

The switch is reachable three ways, all showing the same value:

- **Hover the CRANE-LLM toolbar button.** A small panel drops down with the
  checkbox. Tabbing to the button opens it too, and Escape closes it.
- **The sidebar**, which has the same checkbox.
- **The command palette**, *CRANE-LLM: Include Runtime Information*, which you
  can bind to a key.

It is on by default and remembered across reloads.

| Switch | What goes into the prompt |
|---|---|
| on (default) | executed cells, runtime state of the names the target cell uses, target cell |
| off | executed cells and target cell only |

Turning it off asks the model to predict from the code alone, which is the
comparison the approach is built against. Nothing else changes: the same cells
and the same target cell are sent either way, and the system prompt switches to
a variant that does not tell the model to expect runtime information it is not
being given.

Turning it off is also worth trying when a prompt is too large, since the
runtime section is usually the biggest part.

The target cell is never executed, and predictions are not saved into the
`.ipynb`.

## From Python

```python
%load_ext nb_extension
```

```python
%%crane_llm
model.fit(x_train, y_train)
```

The cell body is analysed, not executed. Arguments are optional:
`--no-runinfo` builds the prompt from the executed cells alone, and anything
else is read as a model name, so `%%crane_llm gpt-5-mini --no-runinfo` works.
This path renders an ipywidgets panel, so it needs the `widgets` extra; the
toolbar button does not.

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
| `widgets` | the ipywidgets panel used by the `%%crane_llm` magic |
| `build` | `jupyter-builder`, needed to rebuild the frontend |
| `experiments` | the batch pipeline in `llms/llm_executor.py` |

## B. Development versus regular install, and how to switch

A regular install puts a *copy* of the code in `site-packages`. Whether your
edits take effect then depends on where the notebook file sits, because a
kernel puts its own working directory first on the import path. A notebook in
the repository root picks up your source; one in any other directory picks up
the stale copy. That ambiguity is why the two modes must not be mixed.

To move from a regular install back to a development install:

```bash
conda activate crane
python -m pip uninstall -y crane_llm
python -m pip install -e ".[widgets,build]"
cd nb_extension && jlpm build && cd ..
jupyter labextension develop . --overwrite
```

`pip uninstall` removes the bundle files it installed, which is why `jlpm build`
is repeated before re-linking. On Windows the last command needs symlink
permission; see [section 1.2.1](#121-if-the-last-command-fails-on-windows).

## C. Pinned dependencies

`package.json` pins three transitive packages under `resolutions`. Do not
remove them without checking that the build still runs.

| Pin | Reason |
|---|---|
| `@rspack/core` `2.0.2` | the bundler version `@jupyter/builder` 1.x expects |
| `glob` `13.0.5` | 13.0.6 ships an empty `dist/commonjs` while its exports map points into it, so `require('glob')` throws and `@jupyter/builder` cannot start |
| `fast-uri` `3.1.3` | 3.1.8 imports `serializePathEncoding` from a module that does not export it, so constructing the `Ajv` instance inside `@jupyter/builder` throws |

Both `glob` and `fast-uri` reached the build through floating version ranges in
packages we do not control, so these pins are the only thing keeping a broken
upstream release out.

`scripts/clean.mjs` exists for the same reason. It replaced the `rimraf`
devDependency, which pulled `glob` in through a floating range.

## C2. The post-build repair step

`build:labextension` runs `scripts/fix-build-load.mjs` afterwards. This is not
optional on Windows.

`@jupyter/builder` finds the bundle it just wrote with
`glob.sync(path.join(staticPath, 'remoteEntry.*.js'))`. On Windows `path.join`
produces backslashes, and glob 9 and later treat a backslash as an escape
character rather than a separator, so the pattern matches nothing. The builder
then records an empty file name and writes `load: path.join('static', '')`,
which is the bare string `"static"`.

JupyterLab resolves that to a directory instead of a module. The extension is
listed by `jupyter labextension list` and appears in the page config, yet never
loads in the browser, with no error in the server log. The repair step rewrites
`load` to the real `static/remoteEntry.<hash>.js`. You can check it by hand:

```bash
node -e "console.log(require('./labextension/package.json').jupyterlab._build.load)"
```

## D. Architecture

| File | Role |
|---|---|
| `src/index.ts` | toolbar button, sidebar, per-cell response boxes |
| `api.py` | kernel-facing entry points; the frontend contract |
| `extension.py` | coordinates tracking, prompt building and the LLM call |
| `ipython_hooks.py` | records successfully executed cells from the kernel |
| `session_state.py` | one ledger entry per cell, not per execution |
| `prompt_builder.py` | assembles the CRANE prompt |
| `runinfo.py` | live-namespace runtime summary |
| `cell_filter.py` | excludes the extension's own helper cells |
| `llm_client.py` | OpenAI Responses API client |

The frontend talks to the backend by running a short snippet in the user's
kernel and reading a delimited JSON payload back off stdout. `api.py` is
therefore a contract: renaming things there breaks the button.

Runtime summarisation is shared with the offline pipeline through
`runinfo_parser/runtime_summary.py`, and retry handling through
`llms/retry.py`, so the extension and the batch experiments cannot drift apart.

The executed-cell ledger is keyed on the notebook's own cell id, which the
kernel receives with every execute request. Re-running a cell replaces its
entry rather than appending a second one, cells that raised are excluded, and
the cell under analysis is never listed as already executed.

Lower-level entry points, for use from a notebook cell:

| Function | Purpose |
|---|---|
| `get_prompt(source, cell_id=..., include_runinfo=True)` | the assembled prompt, no LLM call |
| `run_crane_llm(source, cell_id=..., include_runinfo=True)` | prompt and response |
| `get_live_runinfo_json(target_code)` | runtime summary of the namespace |
| `reload_crane_llm()` | reload the backend in a live kernel |

## E. Validating a change

```bash
python -m py_compile setup.py nb_extension/*.py
python -m nb_extension.smoke_test
cd nb_extension && jlpm build && cd ..
```

`smoke_test.py` covers prompt assembly, the executed-cell ledger, cells that
cannot be parsed, and hostile kernel namespaces. It makes no LLM call.

## F. Troubleshooting

**The toolbar button or sidebar is missing.** Check `jupyter labextension list`
for `crane-llm-jlab`, and that it appears under the first heading rather than
under *Other labextensions*. Then hard-refresh the tab.

**Backend edits have no effect.** You are probably running a regular install,
so the kernel is importing the copy in `site-packages`. Check with
`import nb_extension; print(nb_extension.__file__)` and see appendix B.

**Frontend edits have no effect.** Rarely a cache problem, because the bundle
filename carries a content hash. Far more often JupyterLab is loading the
extension from somewhere other than the directory `jlpm build` writes to. Check
`jupyter labextension list` and see the "Build recommended" entry below.

**The server logs "Build recommended" and "crane-llm-jlab content changed".**
The extension has been registered the old way, as a *source* extension, in
addition to or instead of the prebuilt way. JupyterLab then compiles `lib/`
into its own application bundle and ignores `labextension/` entirely, so
`jlpm build` cannot take effect and every start warns that a rebuild is due.

Confirm it by looking at where `jupyter labextension list` puts the extension.
Under the first heading, next to the `labextensions` path, is correct. Under
*Other labextensions (built into JupyterLab)* or *local extensions* is the
legacy registration. `share/jupyter/lab/settings/build_config.json` will also
have a `local_extensions` entry for it.

This is a **one-time repair**, not something to repeat per change. Once the
extension is registered as prebuilt again, the everyday loop goes back to being
just `jlpm build` (see [section 2](#2-after-changing-frontend-code)).

Stop the Jupyter server, then:

```bash
jupyter labextension uninstall crane-llm-jlab --no-build   # once: drop the legacy entry
jupyter lab clean                                          # once: purge the compiled app dir
cd nb_extension && jlpm build && cd ..                     # the only recurring step
jupyter labextension develop . --overwrite                 # once: register it the modern way
jupyter labextension list                                  # check only
```

Skip the fourth command if the bundle is already linked. On Windows it needs
symlink permission; see
[section 1.2.1](#121-if-the-last-command-fails-on-windows).

Whichever way it is linked, the link keeps pointing at your working copy, so
every later `jlpm build` is picked up with nothing to re-register. You only
redo this step if the link is removed, which `pip uninstall crane_llm` does.

Plain `jupyter lab clean` removes only the staging directory and any source
extensions compiled into the app directory. Prebuilt extensions live outside it
and are unaffected.

**Do not add `--static` or `--all`.** Despite the name, `<app-dir>/static` is
not a user build: the `jupyterlab` wheel installs it there, 295 files of it.
Deleting it leaves the server serving `JupyterLab application assets not found`
on every page. If you do it by accident, restore the files with:

```bash
python -m pip install --force-reinstall --no-deps jupyterlab==4.5.9
```

Never run `jupyter labextension install` or `jupyter labextension link` on this
project. Those are the legacy commands that create this state.

**`jupyter labextension develop` fails with `OSError: Symlinks can be activated
on Windows 10 ... 'Developer Mode'`.** Grant the privilege and re-run it, or use
a regular install; see
[section 1.2.1](#121-if-the-last-command-fails-on-windows). Do not substitute a
directory junction, which is served as 403 by current versions.

**"The kernel did not return a CRANE-LLM payload".** The backend is not
importable from the kernel. Confirm with `import nb_extension.api` in a cell.

**A build fails inside `node_modules` with `MODULE_NOT_FOUND` or a `TypeError`
from a package you never installed.** Yarn can leave an extracted package that
does not match `yarn.lock`, so the version on disk is not the version that was
pinned. Force a clean tree:

```bash
cd nb_extension
node scripts/clean.mjs node_modules .yarn/install-state.gz
jlpm install
```

Check the result before rebuilding, for example
`node -e "console.log(require('glob/package.json').version)"`.

**Authentication errors.** Set `OPENAI_API_KEY` in the environment or in the
repository `.env`.

**"The model hit the output token limit".** Raise `max_output_tokens` in
`llms/config_llms.py`. Reasoning tokens count against that budget.
