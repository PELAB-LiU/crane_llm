# CRANE-LLM

**Predict whether an ML notebook cell will crash — before you run it.**

CRANE-LLM is a JupyterLab extension. Select a cell, click a button, and it tells you whether running that cell is likely to raise, and which earlier cell may have caused the problem.  It does not read your code alone: it also inspects the **live state of your kernel**. The cell you check is never executed.

- **Built-in checks** find crashes that are certain, checking against built-in rules with the live kernel state. They need no LLM and no API key.
- **An LLM** predicts crashes in any target cell you select, using the executed code and a summary of the variables in the kernel state it uses.
- **Origin of the crash** points to the earlier cells that left a variable in the state that makes this cell crash.
- **Check cells before they run** can stop every cell from running if the built-in checks find that it will crash.

On Kaggle, Colab and anywhere else a notebook runs Python, the same checks are available as the `%%crane_llm` cell magic.

## Demo video

A five-minute walkthrough of the [demo notebook](./cranellm_demo/):

https://github.com/user-attachments/assets/7dfc82b4-2444-4448-bc6b-2af57091c988

## Try it

The [demo notebook](./cranellm_demo/) has six target cells to check and shows each part of CRANE-LLM. Three ways to run it:

| | What you need | Setup |
|---|---|---|
| [Docker](#with-docker) | Docker | one command |
| [Your own JupyterLab](#in-your-own-jupyterlab) | Python 3.9+ | `pip install` |
| [Kaggle](#on-kaggle) | a web browser and a Kaggle account | none |

Scenes 1, 5 and 6 of the demo use only the built-in checks and work without an API key. Scenes 2 to 4 ask the LLM, so they need a key, for OpenAI or any [OpenAI-compatible provider](./README_PYPI.md#other-providers). The [demo README](./cranellm_demo/README.md) describes what each scene shows.

### With Docker

The image contains JupyterLab, CRANE-LLM and the demo notebook, at fixed versions, for Linux, macOS and Windows on Intel/AMD and ARM. With [Docker](https://docs.docker.com/get-docker/) installed and running:

```bash
docker run --rm -p 127.0.0.1:8888:8888 -e CRANE_LLM_API_KEY=sk-... ghcr.io/yarinamomo/crane-llm-demo:1.0.1
```

Open <http://localhost:8888>, which opens the demo notebook. Leave out `-e CRANE_LLM_API_KEY=...` to try the built-in checks only. The first run downloads about 250 MB; stop it with Ctrl+C.

### In your own JupyterLab

```bash
pip install "crane-llm[lab]"
jupyter lab
```

Set an API key once, in any notebook cell:

```python
import crane_llm
crane_llm.set_api_key("sk-...")
```

Then run a few cells, select the cell you want to check, and click **CRANE-LLM** in the toolbar. To run the demo this way, download the [demo folder](./cranellm_demo/) and open its notebook.

### On Kaggle

Open the [demo on Kaggle](https://www.kaggle.com/code/yarinawang/crane-llm-demo) and click **Copy & Edit**. Kaggle cannot load the toolbar button, so it uses the `%%crane_llm` cell magic instead. The notebook installs CRANE-LLM, which needs **Internet** switched on in the notebook's settings and a phone-verified Kaggle account. For the LLM scenes, store your key as a Kaggle secret named `CRANE_LLM_API_KEY` (**Add-ons → Secrets**).

The [user guide](./README_PYPI.md) covers the rest: Kaggle and Colab, other providers and local models, the options, checking every cell before it runs, the full list of built-in checks, and what is sent to the LLM model.

## Documentation

| If you want to | Read |
|---|---|
| install and use CRANE-LLM | [the user guide](./README_PYPI.md), which is also the [PyPI project page](https://pypi.org/project/crane-llm/) |
| see which crashes are detected for certain | [Built-in checks](./README_PYPI.md#built-in-checks) in the user guide |
| see what is sent to the model | [Runtime information](./README_PYPI.md#runtime-information) in the user guide |
| run the demo step by step | [the demo README](./cranellm_demo/README.md) |
| change or build the extension | [the developer guide](./crane_llm/nb_extension/README.md) |
| reproduce the paper's experiments | [PAPER.md](./PAPER.md) |
| publish a release or the demo image | [RELEASING.md](./RELEASING.md) |

## Paper

CRANE-LLM is the tool from our SCAM 2026 paper *[CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks](https://arxiv.org/abs/2602.18537)*. The artefacts exactly as published are on the frozen [`paper_scam26`](../../tree/paper_scam26) branch; development continues on this one, and [PAPER.md](./PAPER.md) describes the experiment pipeline it still contains.

## License

BSD 3-Clause. See [LICENSE](./LICENSE).
