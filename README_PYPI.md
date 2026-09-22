# CRANE-LLM

Predict whether a ML notebook cell will crash — before you run it.

CRANE-LLM is a JupyterLab 4 / Notebook 7 extension. Select a cell, click a button, and it tells you whether executing that cell is likely to raise. What makes the prediction useful is that it does not read your code alone: it also inspects the **live state of your kernel**, so it knows the actual shape of the array, the dtype of the column and the length of the list the cell is about to use. The target cell is never executed.

This is the tool from the paper *[CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks](https://arxiv.org/abs/2602.18537)*.

## Install

```bash
pip install crane-llm
jupyter lab
```

That is all — the package ships the compiled frontend, so there is nothing to build and no `jupyter labextension` command to run. Check it registered:

```bash
jupyter labextension list        # expect: crane-llm-jlab <version> enabled ok
```

## Set up a model

You need an API key for whichever model you want to use. One line in any notebook cell, remembered across sessions:

```python
import crane_llm
crane_llm.set_api_key("sk-...")
```

That writes `~/.crane_llm/config.json`. Settings are resolved in this order, and the first one found wins:

1. an explicit argument, e.g. `%%crane_llm gpt-5-mini`
2. the `CRANE_LLM_API_KEY`, `CRANE_LLM_MODEL` and `CRANE_LLM_BASE_URL` environment variables
3. the provider's own variables, `OPENAI_API_KEY` and `OPENAI_BASE_URL`
4. `~/.crane_llm/config.json`

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

There is a cell magic as well, which needs the `widgets` extra (`pip install crane-llm[widgets]`):

```python
%load_ext crane_llm
```

```python
%%crane_llm
model.fit(x_train, y_train)
```

The cell body is analysed, not executed.

## Requirements and what is not included

Python 3.9+, and JupyterLab 4 which is installed as a dependency. Two things are deliberately left out:

- **Notebook 7.** Add `pip install notebook` if you want that interface rather than JupyterLab.
- **The ML stack.** Runtime summarisation understands pandas, numpy, torch, scikit-learn and TensorFlow objects when those packages are present in your kernel, and quietly skips them when they are not. Your own notebooks will already have brought whichever ones they use.

## Links

- Source, issues and full documentation:
  <https://github.com/yarinamomo/crane_llm>
- Releases, including installable wheels:
  <https://github.com/yarinamomo/crane_llm/releases>
- Paper artefacts, datasets and reproduction instructions:
  <https://github.com/yarinamomo/crane_llm/blob/HEAD/PAPER.md>

## License

BSD 3-Clause.
