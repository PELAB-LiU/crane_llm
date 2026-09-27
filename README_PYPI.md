# CRANE-LLM

Predict whether a ML notebook cell will crash — before you run it.

CRANE-LLM is a JupyterLab 4 / Notebook 7 extension. Select a cell, click a button, and it tells you whether executing that cell is likely to raise. What makes the prediction useful is that it does not read your code alone: it also inspects the **live state of your kernel**, so it knows the actual shape of the array, the dtype of the column and the length of the list the cell is about to use. The target cell is never executed.

This is the tool from the paper *[CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks](https://arxiv.org/abs/2602.18537)*.

CRANE-LLM works in two ways:

- **In JupyterLab**, as a toolbar button with a sidebar. This is the full experience.
- **Anywhere else a notebook runs Python** — Kaggle, Colab, VS Code, classic Notebook — as the `%%crane_llm` cell magic. See *On Kaggle and Colab* below.

## Install

```bash
pip install "crane-llm[lab]"
jupyter lab
```

The `[lab]` part installs JupyterLab 4 alongside; leave it out if the environment already has JupyterLab 4: `pip install "crane-llm"`. Check it registered:

```bash
jupyter labextension list        # expect: crane-llm-jlab <version> enabled ok
```

## Set up a model

You need an API key for whichever model you want to use. One line in any notebook cell, remembered across sessions:

```python
import crane_llm
crane_llm.set_api_key("sk-...")
```

That writes `~/.crane_llm/config.json`. On Kaggle and Colab, store the key as a notebook secret instead, as described under *On Kaggle and Colab* below.

Settings are resolved in this order, and the first one found wins:

1. an explicit argument, e.g. `%%crane_llm gpt-5-mini`
2. the `CRANE_LLM_API_KEY`, `CRANE_LLM_MODEL` and `CRANE_LLM_BASE_URL` environment variables
3. the provider's own variables, `OPENAI_API_KEY` and `OPENAI_BASE_URL`
4. `~/.crane_llm/config.json`
5. on Kaggle and Colab, a notebook secret named `CRANE_LLM_API_KEY` or `OPENAI_API_KEY`

A `.env` file is not a step of its own. It is read first, from the directory you started Jupyter in or any directory above it, and fills in whichever of those variables the environment does not already define.

### Using a provider other than OpenAI

Any OpenAI-compatible endpoint works by adding a `base_url` and a model name, which covers Claude and Gemini through a gateway, and local models with no account at all:

| Provider | `base_url` | Example model |
|---|---|---|
| OpenAI | *(none needed)* | `gpt-5` |
| OpenRouter (Claude, Gemini, Llama, …) | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-4.5` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.5-flash` |
| Groq | `https://api.groq.com/openai/v1` | `qwen/qwen3-32b` |
| Ollama, local, no key needed | `http://localhost:11434/v1` | `qwen2.5-coder:32b` |

```python
import crane_llm
crane_llm.set_api_key(
    "sk-or-...",
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-sonnet-4.5",
)
```

Or entirely locally:

```python
crane_llm.set_api_key(base_url="http://localhost:11434/v1", model="qwen2.5-coder:32b")
```

With no `base_url` the OpenAI Responses API is used, which is what the paper's experiments used. Setting a `base_url` switches to Chat Completions, because almost no compatible server implements `/responses`.

## Use it

1. Run some cells as usual and let them finish.
2. Select the cell you want to check.
3. Click **CRANE-LLM** in the toolbar, or run *Run CRANE-LLM* from the command palette.

The verdict appears under the cell, and the right sidebar shows the full prompt and the raw response.

| Colour | Meaning |
|---|---|
| red | a crash is predicted |
| green | no crash is predicted |
| amber | the response could not be read as a verdict |
| grey | the prediction has gone stale |

The notebook must be idle. Reading the kernel namespace means running code in the kernel, and a kernel serves requests in order, so with cells still running the answer would describe the state *afterwards* rather than the one you asked about. A prediction goes stale when you edit the cell or run another one.

A checkbox on the toolbar button, in the sidebar and in the command palette turns the runtime information off, which asks the model to predict from the code alone. That is the comparison the approach is built against, and it is also worth trying when a prompt gets too large.

### What the runtime information contains

Only the variables the target cell uses are included: every name it reads that exists in the kernel, plus the attributes and methods it uses on them.

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

**Value summary:** 

A 1-D array, a Series or a flat list is also described by its values: *binary* with the two values, *categorical* with the number of distinct values (listed when there are 5 or fewer), or *continuous* with its minimum and maximum.

Some of this is your data itself: whole strings, a few values per column, dictionary entries. It is sent to the model provider along with your code, so turn runtime information off for notebooks whose data must not leave your machine. Collecting it does not change your variables.

### The cell magic

Where the toolbar button is not available, put the code you want to check in a cell under `%%crane_llm`:

```python
%load_ext crane_llm
```

```python
%%crane_llm
model.fit(x_train, y_train)
```

The cell body is analysed, not executed. The verdict appears as the cell's output, in the same colours as above, with the prompt and raw response folded under *Prompt and raw response*. It turns grey once you run any other cell, because the kernel state it was based on may have changed; checking another cell with `%%crane_llm` does not count, since nothing is executed.

`%%crane_llm --no-runinfo` turns runtime information off, and a model name overrides the configured one, as in `%%crane_llm gpt-5-mini`.

## On Kaggle and Colab

Hosted notebooks cannot load JupyterLab extensions, so there is no toolbar button or sidebar; the cell magic does the same job.

**1. Store your API key as a secret**, once per account, so that it never appears in the notebook:

- *Kaggle:* in the notebook editor, **Add-ons → Secrets → Add a new secret**, with the label `CRANE_LLM_API_KEY` and your key as the value. Tick the checkbox next to it in each notebook that should use it.
- *Colab:* the key icon in the left sidebar, a secret named `CRANE_LLM_API_KEY`, with **Notebook access** switched on.

CRANE-LLM reads the secret itself; there is no setup cell to write.

**2. Install and load it** in the notebook. On Kaggle, first switch **Internet** on in the notebook's settings panel, which requires a phone-verified account.

```python
%pip install crane-llm
%load_ext crane_llm
```

**3. Run your notebook as usual, then check a cell** by copying its code under `%%crane_llm`:

```python
%%crane_llm
# your code (in the target cell)
```

Hosted sessions start from a fresh image each time, so the `%pip install` cell has to be run again in every new session. Competitions that require Internet to be off cannot use CRANE-LLM, since both the install and the model call need it.

## Requirements and what is not included

Python 3.9+. The toolbar button needs JupyterLab 4, which `crane-llm[lab]` installs; the cell magic needs only IPython. Two things are deliberately left out:

- **Notebook 7.** Add `pip install notebook` if you want that interface rather than JupyterLab.
- **The ML stack.** Runtime summarisation understands *pandas, numpy, torch, scikit-learn, TensorFlow* objects when those packages are present in your kernel, and quietly skips them when they are not. Your own notebooks will already have brought whichever ones they use.

## Links

- Source, issues and full documentation:
  <https://github.com/yarinamomo/crane_llm>
- Releases, including installable wheels:
  <https://github.com/yarinamomo/crane_llm/releases>
- Paper artefacts, datasets and reproduction instructions:
  <https://github.com/yarinamomo/crane_llm/blob/HEAD/PAPER.md>

## License

BSD 3-Clause.
