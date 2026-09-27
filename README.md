# CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks

**Predict whether an ML notebook cell will crash — before you run it.**

This is the official repository for our paper "*[CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks](https://arxiv.org/abs/2602.18537)*". In this paper, we propose CRANE-LLM, a novel approach that prompts LLMs with static code and runtime information extracted from the notebook kernel state to enhance their prediction and explanation of ML notebook crashes.

The repository holds two things: the JupyterLab extension that anyone can install and use, and the artefacts that reproduce the paper.

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

Kaggle and Colab cannot load the toolbar button. There, you use the `%%crane_llm` cell magic instead and keep the key in a notebook secret; the [user guide](./README_PYPI.md) has the steps. It also covers providers other than OpenAI, where the extension looks for its settings, and the runtime-information switch.

## Documentation

| If you want to | Read |
|---|---|
| use the extension | [README_PYPI.md](./README_PYPI.md) — the user guide, which is also the PyPI project page |
| change or build the extension | [crane_llm/nb_extension/README.md](./crane_llm/nb_extension/README.md) |
| reproduce the paper | [PAPER.md](./PAPER.md) |

## License

This project is licensed under the terms of the BSD 3-Clause License.
