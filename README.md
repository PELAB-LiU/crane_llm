# CRANE-LLM

**Predict whether an ML notebook cell will crash — before you run it.**

CRANE-LLM is a JupyterLab extension. Select a cell, click a button, and it tells you whether running that cell is likely to raise, and which earlier cell may have caused the problem. It combines built-in checks, which find crashes that are certain from the live kernel state, with an LLM that sees your code together with a summary of the variables the cell uses. The cell you check is never executed.

On Kaggle, Colab and anywhere else a notebook runs Python, the same checks are available as the `%%crane_llm` cell magic.

## Demo video

A five-minute walkthrough of the [demo notebook](./cranellm_demo/):

https://github.com/user-attachments/assets/7dfc82b4-2444-4448-bc6b-2af57091c988

## Quick start

```bash
pip install "crane-llm[lab]"
jupyter lab
```

Set an API key once, in any notebook cell:

```python
import crane_llm
crane_llm.set_api_key("sk-...")
```

Then run a few cells, select the cell you want to check, and click **CRANE-LLM** in the toolbar.

To see every part of CRANE-LLM at work on a ready-made notebook, open the [demo](./cranellm_demo/). It is also [on Kaggle](https://www.kaggle.com/code/yarinawang/crane-llm-demo), using the cell magic.

The [user guide](./README_PYPI.md) or the [PyPI project page](https://pypi.org/project/crane-llm/) covers the rest: Kaggle and Colab, providers other than OpenAI and local models, the options, checking every cell before it runs, the full list of built-in checks, and what is sent to the model.

## Documentation

| If you want to | Read |
|---|---|
| try CRANE-LLM on an example notebook | [the demo](./cranellm_demo/), or [the same demo on Kaggle](https://www.kaggle.com/code/yarinawang/crane-llm-demo) |
| install and use CRANE-LLM | [the user guide](./README_PYPI.md), which is also the [PyPI project page](https://pypi.org/project/crane-llm/) |
| see which crashes are detected for certain | [Built-in checks](./README_PYPI.md#built-in-checks) in the user guide |
| see what is sent to the model | [Runtime information](./README_PYPI.md#runtime-information) in the user guide |
| change or build the extension | [the developer guide](./crane_llm/nb_extension/README.md) |
| reproduce the paper's experiments | [PAPER.md](./PAPER.md) |
| publish a release | [RELEASING.md](./RELEASING.md) |

## Paper

CRANE-LLM is the tool from our SCAM 2026 paper *[CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks](https://arxiv.org/abs/2602.18537)*. The artefacts exactly as published are on the frozen [`paper_scam26`](../../tree/paper_scam26) branch; development continues on this one, and [PAPER.md](./PAPER.md) describes the experiment pipeline it still contains.

## License

BSD 3-Clause. See [LICENSE](./LICENSE).
