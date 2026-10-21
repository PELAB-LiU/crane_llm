# CRANE-LLM

Predict whether an ML notebook cell will crash — before you run it.

Select a cell, click a button, and CRANE-LLM tells you whether running it is likely to raise, and which earlier cell caused the problem. It does not read your code alone: it also inspects the **live state of your kernel**, so it knows the actual shape of the array, the dtype of the column and the length of the list the cell is about to use. The cell you check is never executed.

Each verdict comes from one of two sources:

- **Built-in checks** find crashes that are certain from the kernel state, such as an undefined name, a missing column or arrays whose shapes do not fit. They run inside your kernel, need no API key and send nothing anywhere.
- **An LLM** predicts crashes in any target cell you select, using the executed code and a summary of the variables in the kernel state it uses.

This is the tool from the paper *[CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks](https://arxiv.org/abs/2602.18537)*.

**Contents:** [Install](#install) · [Set up a model](#set-up-a-model) · [Check a cell](#check-a-cell) · [Try the demo](#try-the-demo) · [Reading the result](#reading-the-result) · [Options](#options) · [Checking every cell before it runs](#checking-every-cell-before-it-runs) · [Privacy](#what-is-sent-to-the-model) · [Reference](#reference)

## Install

### In JupyterLab

```bash
pip install "crane-llm[lab]"
jupyter lab
```

The `[lab]` part installs JupyterLab 4 alongside. If the environment already has JupyterLab 4, `pip install crane-llm` is enough. To check that the extension is registered:

```bash
jupyter labextension list        # expect: crane-llm-jlab <version> enabled ok
```

This gives you a **CRANE-LLM** button in the notebook toolbar and a CRANE-LLM panel in the right sidebar. For the Notebook 7 interface instead of JupyterLab, also `pip install notebook`.

### On Kaggle, Colab and other notebook hosts

Hosted notebooks cannot load JupyterLab extensions, so there is no toolbar button. The `%%crane_llm` cell magic does the same job, and also works in VS Code and classic Notebook. In the first cell of the notebook:

```python
%pip install crane-llm
%load_ext crane_llm
```

On Kaggle, first switch **Internet** on in the notebook's settings panel, which requires a phone-verified account. Hosted sessions start from a fresh image, so this cell has to be run again in every new session. Competitions that require Internet to be off cannot use CRANE-LLM, since the install needs it.

### Requirements

Python 3.9 or later. The toolbar button needs JupyterLab 4; the cell magic needs only IPython. CRANE-LLM understands pandas, NumPy, PyTorch, scikit-learn and TensorFlow objects when those packages are present in your kernel, and does not install them itself.

## Set up a model

You need an API key for the model you want to use, unless you only use the built-in checks or run a local model.

**On your own machine**, run this once in any notebook cell. The key is saved in `~/.crane_llm/config.json` and remembered across sessions:

```python
import crane_llm
crane_llm.set_api_key("sk-...")
```

**On Kaggle and Colab**, store the key as a notebook secret named `CRANE_LLM_API_KEY`, so that it never appears in the notebook. CRANE-LLM reads the secret itself.

- *Kaggle:* in the notebook editor, **Add-ons → Secrets → Add a new secret**. Tick the checkbox next to it in each notebook that should use it.
- *Colab:* the key icon in the left sidebar, with **Notebook access** switched on.

### Other providers

Any OpenAI-compatible endpoint works by adding a `base_url` and a model name. This covers Claude and Gemini through a gateway, and local models that need no account:

| Provider | `base_url` | Example model |
|---|---|---|
| OpenAI | *(none needed)* | `gpt-5`(default) |
| OpenRouter (Claude, Gemini, Llama, …) | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-4.5` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.5-flash` |
| Groq | `https://api.groq.com/openai/v1` | `qwen/qwen3-32b` |
| Ollama (local, no key) | `http://localhost:11434/v1` | `qwen2.5-coder:32b` |

```python
import crane_llm
crane_llm.set_api_key(
    "sk-or-...",
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-sonnet-4.5",
)

# or entirely locally:
crane_llm.set_api_key(base_url="http://localhost:11434/v1", model="qwen2.5-coder:32b")
```

With a `base_url`, requests use the Chat Completions API, which every OpenAI-compatible server supports.

### Where settings are read from

The first setting found wins:

1. an explicit argument, such as `%%crane_llm gpt-5`
2. the `CRANE_LLM_API_KEY`, `CRANE_LLM_MODEL` and `CRANE_LLM_BASE_URL` environment variables
3. the provider's own variables, `OPENAI_API_KEY` and `OPENAI_BASE_URL`
4. `~/.crane_llm/config.json`, written by `set_api_key`
5. on Kaggle and Colab, a notebook secret named `CRANE_LLM_API_KEY` or `OPENAI_API_KEY`

A `.env` file in the directory you started Jupyter in, or any directory above it, is read first and fills in whichever of those environment variables are not already set.

## Check a cell

First run your notebook as usual and let the cells finish. CRANE-LLM reads the kernel's current state, so it refuses to start while a cell is still running.

**In JupyterLab:** select the cell you want to check and click **CRANE-LLM** in the toolbar (top-right), or run *Run CRANE-LLM* from the command palette. The verdict appears under the cell, and the sidebar shows each step as it happens, then the full prompt and the raw response.

**With the cell magic:** copy the code you want to check into a new cell under `%%crane_llm`:

```python
%%crane_llm
model.fit(x_train, y_train)
```

The cell body is analysed, not executed. The verdict is the cell's output, with the prompt and raw response folded underneath.

### Try the demo

[![CRANE-LLM in JupyterLab: a certain AttributeError found by the built-in checker, with the origin of the crash. Click to watch the demo video.](https://raw.githubusercontent.com/yarinamomo/crane_llm/HEAD/cranellm_demo/crane-llm-demo_screenshot.png)](https://github.com/yarinamomo/crane_llm#demo-video)

*Click the screenshot to watch the five-minute demo video on GitHub.*

The repository has a [demo notebook](https://github.com/yarinamomo/crane_llm/tree/HEAD/cranellm_demo) to try this on: a short regression notebook with six cells to check, five of which crash. It walks through the built-in checks, the LLM prediction, runtime information, the origin of a crash, and checking cells before they run. Download the folder, open the notebook in JupyterLab and follow its README.

The same demo is also [on Kaggle](https://www.kaggle.com/code/yarinawang/crane-llm-demo), using the cell magic. Click **Copy & Edit** to run it in your own Kaggle account.

## Reading the result

The verdict is shown in a colour:

| Colour | Meaning |
|---|---|
| red | a crash is certain or predicted |
| green | no crash is predicted |
| blue | only the built-in checks ran and found nothing certain. This is not a prediction that the cell is safe |
| amber | the model's response could not be read as a verdict; the raw text is still shown |
| grey | the result is stale: you have edited the cell or run another one since |

A badge next to the verdict says where it came from:

- **Built-in check · certain** — CRANE-LLM found the crash itself and did not call the model. The cell will raise when it runs.
- **LLM prediction · *model*** — the model judged the cell. This is a prediction, and it can be wrong.

### Origin of the crash

The cell that crashes is rarely where the mistake was made. When a crash is found or predicted, the result lists under **Origin of the crash** the cells that gave the variables involved their current state:

- the cell that last **assigned** the variable, and the line that did it,
- every cell that **modified** it since, for example by dropping columns in place or fitting a model,
- for a variable that does not exist, the cells that would **define** it and why they did not: not run yet, or raised before reaching the assignment. This one needs the whole notebook, so it is shown in JupyterLab only.

In JupyterLab, click an entry to jump to its cell. Those cells are outlined in violet with a short note until the result goes stale.

CRANE-LLM records the order cells actually ran in from the moment the kernel starts, or from `%load_ext crane_llm` with the cell magic, so load it early. Cells run before that are known from their code only, and are marked as such.

## Options

| Option | JupyterLab | Cell magic | Default |
|---|---|---|---|
| Use the LLM or only the built-in checker | *Use the LLM* switch | `--no-llm` turns it off | on |
| Include runtime information | *Include runtime information* switch | `--no-runinfo` turns it off | on |
| Model | settings, see [above](#where-settings-are-read-from) | a model name, such as `gpt-5-mini` | from settings |

In JupyterLab, the switches are in a panel that opens when you hover over the toolbar button, in the sidebar, and in the command palette. They are remembered across sessions. With the cell magic, a flag applies only to that one check, and flags combine: `%%crane_llm gpt-5-mini --no-runinfo`.

- **Use the LLM.** Turned off, only the built-in checks run. Nothing is sent to any model and no API key is needed.
- **Include runtime information.** Turned off, the model predicts from the code alone, and the built-in checks are skipped too, since they read the kernel state. Turning it off is also worth trying when a prompt is too large.

### Scan limit

Some built-in checks read every value of your data, such as `astype(int)` or looking for NaN in scikit-learn input. They are skipped for data with more than a million values, so that a check never takes noticeably long. To change the limit:

```python
import crane_llm
crane_llm.set_scan_limit(5_000_000)   # 0 turns these checks off; None restores the default
```

or set the `CRANE_LLM_SCAN_LIMIT` environment variable.

## Checking every cell before it runs

The built-in checks can also run by themselves, on every cell, just before it runs. When they find that a cell will crash, CRANE-LLM does not run it: the cell shows why, with the origin of the crash, and ends with a `CrashPrevented` error. No line of the cell runs. Like any error, this stops *Run All* at that cell.

Only the built-in checks are used, never the model, so nothing is sent anywhere and no API key is needed. A cell in which the checks find nothing runs as usual, which does not mean it is safe.

| | JupyterLab | Cell magic |
|---|---|---|
| turn on | *Check cells before they run* switch | `%crane_llm guard on` |
| turn off | the same switch | `%crane_llm guard off` |
| lasts | remembered, for every notebook | until the kernel restarts |

`%crane_llm guard` on its own says whether it is on. It is off by default.

To run a stopped cell anyway, for example to see Python's own error, add this line anywhere in it:

```python
# crane: run
```

Cells that start with a cell magic, such as `%%time`, are not checked.

## What is sent to the model

When the LLM is used, the prompt contains the cells already run, the cell to check and, with runtime information on, a summary of the variables that cell uses. Some of that summary is your data itself: whole strings, a few values per column, dictionary entries. Exactly what is included for each kind of object is listed under [Runtime information](#runtime-information) below.

For notebooks whose data must not leave your machine, turn runtime information off, turn the LLM off, or use a local model. Collecting runtime information does not change your variables, and results are not saved into the `.ipynb`.

## Reference

### Built-in checks

A built-in check reports a crash only when it is certain, and never declares a cell safe: when it finds nothing, the model is asked. It reads the cell in the order Python runs it and stops at code whose effect it cannot know, such as a call to one of your own functions or the body of a loop.

Past such code it carries on, but only with what that code cannot have changed: which variables exist, and values that cannot change at all, such as numbers, strings and `None`, under names that code cannot reassign.

| Library | The cell... | Raises |
|---|---|---|
| Python | uses a name that is not defined | `NameError` |
| Python | reads an attribute that does not exist, including on `None` (`df = df.dropna(inplace=True)` leaves `df` as `None`) and APIs a library has removed, such as `DataFrame.append` or `np.float` | `AttributeError` |
| Python | imports a module that is not installed, or a name a module does not have, such as `from sklearn import cross_validation` | `ModuleNotFoundError`, `ImportError` |
| Python | calls a function with arguments its signature does not accept: a missing or extra argument, a positional argument where only keywords are allowed, or a keyword a library has removed, such as `read_csv(error_bad_lines=...)` | `TypeError` |
| Python | calls something that cannot be called, such as `df.shape()` or `df.columns()` | `TypeError` |
| Python | opens a file that does not exist, with `open`, `pd.read_csv`, `read_excel`, `read_parquet` and similar, `np.load` or `os.listdir` | `FileNotFoundError` |
| Python | loops over, or unpacks, something that cannot be iterated, such as `for x in None` | `TypeError` |
| Python | tests membership in something that is not a container (`x in 5`), or orders values that cannot be compared (`None > 0`, `1 < "a"`) | `TypeError` |
| Python | uses a list, dict or set as a dict key | `TypeError` |
| Python | assigns into a tuple, a string or `None` | `TypeError` |
| Python | formats a string with too few or too many arguments (`"{} {}".format(x)`, `"%d %d" % (x,)`) or a format code the value does not support (`f"{1.5:d}"`) | `IndexError`, `KeyError`, `ValueError`, `TypeError` |
| Python | applies `-`, `+` or `~` to a value that does not support it, such as `-"a"` or `-None` | `TypeError` |
| Python | deletes from a tuple or `None`, or deletes a missing key or index (`del d["x"]`, `del lst[5]`) | `TypeError`, `KeyError`, `IndexError` |
| Python | passes `*x` where `x` cannot be iterated, or `**x` where `x` is not a mapping | `TypeError` |
| Python | loops with `for a, b in ...` over items that do not unpack into two, such as the keys of a dict | `ValueError`, `TypeError` |
| Python | slices a list, tuple or string with bounds that are not integers, or a step of zero | `TypeError`, `ValueError` |
| Python | reaches a `raise` outside any `try`, including `raise` of something that is not an exception | the raised exception, `TypeError` |
| Python | reads an environment variable that is not set with `os.environ["NAME"]`, or calls `os.getenv` with a name that is not a string | `KeyError`, `TypeError` |
| Python | instantiates a class that still has abstract methods | `TypeError` |
| Python | reads a missing dict key, or a list or tuple index out of range | `KeyError`, `IndexError` |
| Python | divides by zero, adds incompatible types such as `None + 1`, unpacks the wrong number of values, or cannot be parsed | `ZeroDivisionError`, `TypeError`, `ValueError`, `SyntaxError` |
| pandas | selects, drops, groups, sorts or indexes by a column that does not exist | `KeyError` |
| pandas | uses `.loc` with a row or column label that does not exist | `KeyError` |
| pandas | uses `.iloc` with a position past the end, such as `df.iloc[100]` on a frame of 10 rows | `IndexError` |
| pandas | converts a column with `astype(int)` that holds a value which cannot become an integer, such as the text `"n/a"`, `None`, or NaN | `ValueError`, `IntCastingNaNError`, `TypeError` |
| pandas | assigns a list of the wrong length as a column | `ValueError` |
| pandas | uses `.str`, `.dt` or `.cat` on a column of another type, such as `.str` on numbers | `AttributeError` |
| pandas | adds, multiplies or compares a Series with a list or array of a different length | `ValueError` |
| pandas | merges or joins on a key column one of the frames does not have, or concatenates nothing | `KeyError`, `ValueError` |
| pandas, NumPy | tests a DataFrame, a Series or an array of more than one element for truth: `if df:`, `not arr` | `ValueError` |
| NumPy | multiplies, adds, stacks or reshapes arrays whose shapes do not fit, or indexes past the end | `ValueError`, `IndexError` |
| NumPy | reduces along an axis the array does not have, such as `a.sum(axis=3)` on a 2-D array | `AxisError` |
| NumPy | indexes a plain array with a string (`arr["price"]`), or a list with an array of several values (`items[np.array([0, 2])]`) | `IndexError`, `TypeError` |
| NumPy | converts text that is not a number with `astype(int)`, or passes a function data it rejects for its number of dimensions or its missing values | `ValueError` |
| scikit-learn | calls `predict` on a model that has not been fitted | `NotFittedError` |
| scikit-learn | passes a model a different number of features, or columns with different names or order, than it was fitted on | `ValueError` |
| scikit-learn | fits a model, or calls `train_test_split`, with `X` and `y` of different lengths | `ValueError` |
| scikit-learn | fits a classifier on a continuous target, or scores continuous predictions with a classification metric such as `accuracy_score` | `ValueError` |
| scikit-learn | creates an estimator with an argument it does not take, or calls `train_test_split` with a `test_size` that is neither a fraction nor a count below the number of samples | `TypeError`, `ValueError` |
| scikit-learn | passes data with NaN or infinity to an estimator that does not accept missing values, or text such as `"male"` to one that only takes numbers | `ValueError` |
| scikit-learn | predicts with k-nearest neighbours where `n_neighbors` is larger than the number of samples fitted | `ValueError` |
| matplotlib | shows an array with `plt.imshow` that is not an image: not 2-D, or 3-D without 1, 3 or 4 channels | `TypeError` |
| PyTorch | passes an `nn.Linear` layer input whose last dimension is not its `in_features` | `RuntimeError` |
| PyTorch | multiplies (`@`, `torch.matmul`), adds, concatenates (`torch.cat`) or reshapes (`view`, `reshape`) tensors whose shapes do not fit | `RuntimeError` |
| PyTorch | calls `.numpy()` on a tensor that requires grad, or `.item()` on a tensor of more than one value | `RuntimeError` |
| Keras | reads a metric the training history did not record, or calls a method a model does not have | `KeyError`, `AttributeError` |

### Runtime information

Only the variables the checked cell uses are included: every name it reads that exists in the kernel, plus the attributes and methods it uses on them.

| Object | What the prompt includes |
|---|---|
| `int`, `float`, `str`, `bool` | the value itself, including the full text of a string |
| `list`, `tuple`, `set` | length. A flat list also gets the value summary below |
| `dict` | length, and depending on the contents: the metric names and epoch count of a Keras training history; the keys and data/target shapes of a scikit-learn dataset; the keys of a dict of numbers; otherwise the first 5 entries, with each value shown up to 50 characters |
| NumPy array | shape, dtype, whether it contains NaN, minimum and maximum |
| pandas Series | dtype, length, whether it contains NaN |
| pandas DataFrame | shape, whether it contains NaN, and for each of the first 20 columns: dtype, number of distinct values, and either the minimum and maximum (numeric columns) or up to 5 values, each shortened to 20 characters (other columns) |
| PyTorch tensor | shape, dtype, device, `requires_grad`, whether it contains NaN |
| PyTorch `DataLoader` and `Subset` | number of batches and examples, batch size, the dataset's fields, and the shapes of its first 10 samples and of a batch built from them |
| TensorFlow `tf.data` dataset | its element spec |
| Keras `DirectoryIterator` and `DataFrameIterator` | number of samples and classes, batch size, image shape |
| scikit-learn estimator | class, whether it has been fitted, and once fitted, the number of input features and outputs. A fitted `LabelEncoder` adds its number of classes |
| functions, methods, classes, modules | the type only |

**Value summary.** A 1-D array, a Series or a flat list is also described by its values: *binary* with the two values, *categorical* with the number of distinct values (listed when there are 5 or fewer), or *continuous* with its minimum and maximum.

## Links

- Source code and issues: <https://github.com/PELAB-LiU/crane_llm>
- Releases, including installable wheels: <https://github.com/PELAB-LiU/crane_llm/releases>

## License

BSD 3-Clause.
