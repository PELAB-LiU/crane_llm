# CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks

This is the official repository for our paper "CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks". In this paper, we propose CRANE-LLM, a novel approach that prompts LLMs with static code and runtime information extracted from the notebook kernel state to enhance their prediction and explanation of ML notebook crashes.

## Using the CRANE-LLM notebook extension

CRANE-LLM ships as a JupyterLab 4 / Notebook 7 extension. You do not need to
clone this repository or build anything to use it: the published wheel already
contains the compiled frontend.

### Installing

```bash
pip install crane-llm
jupyter lab
```

Or, to install a specific release directly from GitHub without PyPI. Take the
version from the [releases page](https://github.com/yarinamomo/crane_llm/releases)
and substitute it in both places:

```bash
pip install https://github.com/yarinamomo/crane_llm/releases/download/v0.0.0/crane_llm-0.0.0-py3-none-any.whl
```

Check that it registered:

```bash
jupyter labextension list        # expect: crane-llm-jlab <version> enabled ok
```

Then open a notebook, run a few cells, select the cell you want to check, and
click **CRANE-LLM** in the toolbar. The full walkthrough is in
[Using the extension](./crane_llm/nb_extension/README.md#5-using-the-extension).

Two things the install deliberately leaves out. Add `pip install notebook` if
you want the Notebook 7 interface rather than JupyterLab, and note that the ML
stack is not pulled in: runtime summarisation covers pandas, numpy, torch,
sklearn and TensorFlow objects when those packages are present in your kernel,
and quietly skips them when they are not.

To work on the extension rather than use it, see
[the extension README](./crane_llm/nb_extension/README.md), which covers the
source build. To publish a new version of it, see [RELEASING.md](./RELEASING.md).

### Setting up a model

You need an API key for whichever model you want to use. The quickest way,
which is remembered across sessions, is one line in any notebook cell:

```python
import crane_llm
crane_llm.set_api_key("sk-...")
```

That writes `~/.crane_llm/config.json`. The extension looks for each setting in
this order, and takes the first one it finds:

1. an explicit argument, e.g. `%%crane_llm gpt-5-mini`
2. the `CRANE_LLM_API_KEY`, `CRANE_LLM_MODEL` and `CRANE_LLM_BASE_URL` environment variables
3. the provider's own variables, `OPENAI_API_KEY` and `OPENAI_BASE_URL`
4. `~/.crane_llm/config.json`
5. for the model only, the default in [`config_llms.py`](./crane_llm/llms/config_llms.py)

**A `.env` file is not a step of its own.** Before the lookup runs, any `.env`
in the directory you started Jupyter from, or in a directory above it, is read
and used to fill in whichever of those variables the environment does not
already define. Its values are then found at step 2 or step 3, under whatever
variable name they were written with. Two consequences:

- a variable already present in the environment beats the same name in `.env`,
  because the file never overwrites something already set. This includes a
  persistent variable, such as a Windows user environment variable, which a
  Jupyter kernel inherits without your having exported anything. Keeping the
  same key in both places is fine; just remember that editing only the `.env`
  copy will appear to do nothing
- `OPENAI_API_KEY=...` in `.env` beats a key in `~/.crane_llm/config.json`,
  because it is read at step 3 and the file is step 4

#### Using a model other than OpenAI

Any OpenAI-compatible endpoint works by adding a `base_url` and a model name.
This covers most providers, including Claude and Gemini through a gateway:

| Provider | `base_url` | Example model |
|---|---|---|
| OpenAI | *(none needed)* | `gpt-5` |
| OpenRouter (Claude, Gemini, Llama, …) | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-4.5` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.5-flash` |
| Groq | `https://api.groq.com/openai/v1` | `qwen/qwen3-32b` |
| Ollama, local, no key needed | `http://localhost:11434/v1` | `qwen2.5-coder:32b` |

For example, to use Claude through OpenRouter:

```python
import crane_llm
crane_llm.set_api_key(
    "sk-or-...",
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-sonnet-4.5",
)
```

Or a local model with no account at all:

```python
crane_llm.set_api_key(base_url="http://localhost:11434/v1", model="qwen2.5-coder:32b")
```

With no `base_url`, CRANE-LLM calls OpenAI's Responses API, which is what the
experiments in the paper used. As soon as a `base_url` is set it switches to
Chat Completions, because almost no compatible server implements `/responses`.
Azure OpenAI is the exception: it needs a `base_url` *and* the Responses API,
so set `CRANE_LLM_API_STYLE=responses` there.

## Paper artefacts and reproducibility

The dataset, the models, what every directory in this repository holds, the
experiment scripts and the docker image used to produce the results are
documented separately, in **[PAPER.md](./PAPER.md)**.

## License

This project is licensed under the terms of the BSD 3-Clause License.
