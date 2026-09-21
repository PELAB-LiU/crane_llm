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

Or, to install a specific release directly from GitHub without PyPI:

```bash
pip install https://github.com/PELAB-LiU/crane-llm/releases/download/v0.1.0/crane_llm-0.1.0-py3-none-any.whl
```

Check that it registered:

```bash
jupyter labextension list        # expect: crane-llm-jlab v0.1.0 enabled ok
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
source build.

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
6. the default model in [`config_llms.py`](./crane_llm/llms/config_llms.py)

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

## Repository structure and reproducibility details
Dataset: We use [**Junobench**]((https://huggingface.co/datasets/PELAB-LiU/JunoBench)) dataset in our experiments.

LLMs: LLMs include Gemini (Gemini-2.5-Flash), Qwen (Qwen-2.5-Coder-32B-Instruct), GPT-5.

Repository structure:

All importable Python code lives under the single package
[`crane_llm/`](./crane_llm), which is also what the installable wheel contains.
The scripts at the repository root drive the experiments and are run from a
checkout rather than installed. Generated experiment inputs and outputs stay in
[`llms/`](./llms) at the root, beside the code that produces them.

- [`crane_llm/`](./crane_llm): the installable package
    - [`nb_extension/`](./crane_llm/nb_extension): the JupyterLab extension, documented in [its own README](./crane_llm/nb_extension/README.md)
    - [`runinfo_parser/`](./crane_llm/runinfo_parser): scripts and configuration for *runtime information extraction*
    - [`config_llms.py`](./crane_llm/llms/config_llms.py): configuration for experiments, including prompts, LLM configs, and related input and output paths
    - [`prompt_extractor.py`](./crane_llm/llms/prompt_extractor.py): script for constructing prompts (i.e., `llms_inputs/`)
    - [`llm_executor.py`](./crane_llm/llms/llm_executor.py): script for querying LLMs to generate outputs in `llms_outputs/`
- [`crane-llm.py`](./crane-llm.py): script to run CRANE-LLM given a target notebook
- [`main_LLM.py`](./main_LLM.py): script to run the experiment pipeline, including batch run all notebooks in the dataset for a specific task and experimental setting
- [`main.py`](./main.py): script to run result compilation and analysis
- [`llms`](./llms): LLM experiment inputs and outputs
    - [`llms_inputs/`](./llms/llms_inputs): generated inputs (executed code cells only, executed code cells with runtime information) to the LLMs
    - [`llms_outputs/`](./llms/llms_outputs): generated outputs by the LLMs
        - [`results_raw/`](./llms/llms_outputs/results_raw/): LLM outputs organized into: prefix\_[*LLM*]\_[*experimental setup*]\_[*runtime category ablation setup or API grounding*]
        - [`ground_truth_crash_prediction.xlsx`](./llms/llms_outputs/ground_truth_crash_prediction.xlsx): ground truth labels used for evaluating LLM outputs as well as downstream analysis, provided by JunoBench
- [`results`](./results): evaluated result outcomes and compiled statistics
    - [`results_parsed_detection_and_diagnosis.xlsx`](./results/results_parsed_detection_and_diagnosis.xlsx): CRANE-LLM performance on the joint crash prediction and diagnosis task
        - Sheet "Final_evaluation": Detailed outcomes per LLM per experimental setup. Settings include:
            - code: -RT
            - runinfo: +RT (CRANE-LLM)
        - Sheet "Results_summary": Compiled results and statistics on crash prediction and diagnosis performance of CRANE-LLM
    - [`results_parsed_detection_only.xlsx`](./results/results_parsed_detection_only.xlsx): CRANE-LLM performance on the crash prediction-only task
        - Sheet "Final_evaluation": Detailed outcomes per LLM per experimental setup including *runtime information category ablation study*, and *API documentation grounding study*. All settings include:
            - code: -RT
            - runinfo: +RT (CRANE-LLM)
            - runinfo_r_v: +RT-S (CRANE-LLM - S), ablated structural runtime information
            - runinfo_s_r: +RT-V (CRANE-LLM - V), ablated value semantics runtime information
            - runinfo_s_v: +RT-R (CRANE-LLM - R), ablated type-level (representation and type semantics) runtime information
            - runinfo_full_doc: +RT+doc (CRANE-LLM + doc), full runtime information with additional API documentation information
        - Sheet "Results_summary": Compiled results and statistics on crash prediction-only performance of CRANE-LLM, including runtime information category ablation study and API documentation grounding study results (and token analysis results)
    - [`runtime_doc_token_analysis.txt`](./results/runtime_doc_token_analysis.txt): statistics of tokens of additional API documentation information, results gained by running script [`token_analysis.py`](./utils/token_analysis.py)
    - [`pairwise_significance_detection_and_diagnosis.json`](./results/pairwise_significance_detection_and_diagnosis.json) and [`pairwise_significance_detection_only.json`](./results/pairwise_significance_detection_only.json): statistical test results of the joint crash prediction and diagnosis task and the crash prediction-only task, the statistics tests are ran by script [statistical_test.py](./utils/statistical_test.py)
    - [`cohens_kappa_human_validation.txt`](./results/cohens_kappa_human_validation.txt): statistics of human evaluation on crash diagnosis outputs
    - [`runtime_recording/`](./results/runtime_recording/): statistics of runtime for prior cell executions and querying CRANE-LLM (when using GPT-5).

## Environment

To ensure full reproducibility, we provide a docker image (digest: sha256:ecb5753d1cdfc9f0d5dfeb59818cde5be5be2f79541c5facf99761393919e171):
```bash
docker pull yarinamomo/crane_env:latest
```
Then run the docker container:
```bash
docker run -v [volumn_mount_windows_path]:/cranellm_env -w /cranellm_env -p 8888:8888 -it yarinamomo/crane_env:latest /bin/bash
```
Then you can attach this environment to **VS Code** "*Dev Containers: Attach to Running Container...*"

For the commercial LLMs used in the experiments (Gemini and GPT-5), please ensure that the API keys are properly set up before running the scripts (for example, set as global environment variable or config in `.env`). Open-source LLMs (Qwen) can be run directly; however, note that execution may take longer depending on the computational resources available.

## License

This project is licensed under the terms of the BSD 3-Clause License.
