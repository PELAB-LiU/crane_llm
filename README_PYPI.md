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
| red | a crash is certain or predicted |
| green | no crash is predicted |
| amber | the response could not be read as a verdict |
| grey | the prediction has gone stale |

A badge next to the verdict says where it came from:

- **Built-in check · certain**: CRANE-LLM found the crash itself, from the live kernel state, and did not call the model. The cell will raise when it runs.
- **LLM prediction · *model***: the model judged the cell. This is a prediction, and it can be wrong.

The notebook must be idle. Reading the kernel namespace means running code in the kernel, and a kernel serves requests in order, so with cells still running the answer would describe the state *afterwards* rather than the one you asked about. A prediction goes stale when you edit the cell or run another one.

Two switches sit on the toolbar button (hover over it), in the sidebar and in the command palette:

- **Use the LLM.** On by default. Turned off, only the built-in checks run and nothing is sent to any model, so no API key is needed. When the checks find no certain crash, the verdict says so in blue: that is not a "no crash" prediction, since the checks only report crashes they are certain of.
- **Include runtime information.** Turned off, the model predicts from the code alone. That is the comparison the approach is built against, and it is also worth trying when a prompt gets too large. With it off, the built-in checks are skipped too, because they read the live kernel state. It has no effect while the LLM is off.

### What the runtime information contains

Only the variables the target cell uses are included: every name it reads that exists in the kernel, plus the attributes and methods it uses on them.

For each object, the prompt describes its type and the facts that crashes usually depend on: the value of a number or string, the shape, dtype and NaN status of an array, the columns of a DataFrame, whether a model has been fitted. What exactly is included for each kind of object is listed under [Runtime information sent to the model](https://github.com/yarinamomo/crane_llm#runtime-information-sent-to-the-model) in the project README.

Some of this is your data itself: whole strings, a few values per column, dictionary entries. It is sent to the model provider along with your code, so turn runtime information off for notebooks whose data must not leave your machine. Collecting it does not change your variables.

### Built-in checks

Before calling the model, CRANE-LLM checks the cell against the live kernel state for crashes that can be detected for certain: an undefined name, a column that does not exist, arrays whose shapes do not fit, a model that has not been fitted, a file that is missing, and many more. When one of these is found, you get the answer immediately, without waiting for the model or spending tokens on it. The full list, by library, is under [Built-in checks](https://github.com/yarinamomo/crane_llm#built-in-checks) in the project README.

A check reports only crashes that are certain. It reads the cell from the top in the order Python runs it. The moment the cell would run code whose effect it cannot know, such as a call to one of your own functions or the body of a loop, an `if` or a `try` block, it stops and leaves the cell to the model. The first line of a `for`, `if`, `while` or `with` is still checked, since it always runs: `for x in None:` and `with open("missing.csv")` are caught, but `df['nope']` inside the loop is not. A check never declares a cell safe: when nothing certain is found, the model is asked exactly as before.

Some checks read every value of your data: `astype(int)`, NaN and text in scikit-learn input, missing values passed to NumPy functions, `.str`/`.dt`/`.cat`, the continuous-target checks, and the classification metrics. They are skipped for data with more than a million values, so that a check never takes noticeably long, and the model is asked instead. Change the limit once with:

```python
import crane_llm
crane_llm.set_scan_limit(5_000_000)   # 0 turns these checks off; None restores the default
```

or set the `CRANE_LLM_SCAN_LIMIT` environment variable.

While a check runs, the sidebar and a box under the cell list each step as it happens: the built-in checker, then, if it found nothing certain, building the prompt and waiting for the model. A verdict from the model also says that the checker ran first and found nothing.

The checks run inside your kernel and send nothing anywhere.

### Origin of the crash

The cell that crashes is rarely where the mistake was made. When a crash is found or predicted, CRANE-LLM lists under **Origin of the crash** the cells that gave the variables involved their current state:

- the cell that last **assigned** the variable, and the line that did it,
- every cell that **modified** it since, for example by dropping columns in place or fitting a model,
- for a variable that does not exist, the cells of the notebook that **define it** and why they did not: not run yet, or raised before reaching the assignment. This one needs the whole notebook, so it appears in JupyterLab only, not with the cell magic.

Click an entry to jump to its cell. The cells themselves are outlined with a dashed violet line and carry a short note saying what they did, until the verdict goes stale.

This works from the order the cells actually ran in, which the notebook file does not record. CRANE-LLM records it from the moment the kernel starts, or from `%load_ext crane_llm` with the cell magic. Cells run before that are known from their code only, and are marked as such.

### The cell magic

Where the toolbar button is not available, put the code you want to check in a cell under `%%crane_llm`:

```python
%load_ext crane_llm
```

```python
%%crane_llm
model.fit(x_train, y_train)
```

The cell body is analysed, not executed. The verdict appears as the cell's output, in the same colours and with the same badge as above, followed by the origin of the crash and, folded under *Prompt and raw response*, what was sent to the model. It turns grey once you run any other cell, because the kernel state it was based on may have changed; checking another cell with `%%crane_llm` does not count, since nothing is executed.

`%%crane_llm --no-runinfo` turns runtime information off, `%%crane_llm --no-llm` runs only the built-in checks, and a model name overrides the configured one, as in `%%crane_llm gpt-5-mini`.

## On Kaggle and Colab

Hosted notebooks cannot load JupyterLab extensions, so there is no toolbar button or sidebar; the cell magic does the same job.

**1. Store your API key as a secret**, once per account, so that it never appears in the notebook. Skip this step if you only want the built-in checks, described in step 4.

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

**4. Choose what runs**, with a flag on the first line. There are no switches on hosted notebooks, so the flag applies to that one check only and has to be written again on the next one.

| First line | What happens | API key needed |
|---|---|---|
| `%%crane_llm` | The built-in checks run first. If they find no certain crash, the model is asked, with runtime information. | yes |
| `%%crane_llm --no-llm` | Only the built-in checks run. Nothing is sent to any model. | no |
| `%%crane_llm --no-runinfo` | The model judges the code alone. The built-in checks are skipped, since they read the live kernel state. | yes |

With `--no-llm`, a cell where the checks find nothing gets a blue verdict saying so. That is not a prediction that the cell is safe: the checks only report crashes they are certain of. Adding `--no-runinfo` to `--no-llm` changes nothing, since without the model the checks are all that runs. A model name can be combined with the other flags, as in `%%crane_llm gpt-5-mini --no-runinfo`.

Hosted sessions start from a fresh image each time, so the `%pip install` cell has to be run again in every new session. Competitions that require Internet to be off cannot use CRANE-LLM, since the install needs it, and so does the model call unless you use `--no-llm`.

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
