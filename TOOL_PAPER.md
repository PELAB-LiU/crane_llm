# Tool paper artefacts

Artefacts for our tool demonstration paper on the CRANE-LLM notebook extension. It builds on the [CRANE-LLM paper](./PAPER.md).

If you are here to *use* the notebook extension rather than reproduce the paper evaluation results, see the [main README](./README.md) instead.

## Repository structure

The extension lives in [`crane_llm/nb_extension/`](./crane_llm/nb_extension). The [developer guide](./crane_llm/nb_extension/README.md#4-architecture) describes every file in it; these are the parts behind each feature of the paper:

- The user interface: [`src/index.ts`](./crane_llm/nb_extension/src/index.ts) and [`style/index.css`](./crane_llm/nb_extension/style/index.css), the JupyterLab toolbar button, sidebar, verdict boxes and origin marks
- Support beyond JupyterLab, such as Kaggle: [`magic.py`](./crane_llm/nb_extension/magic.py), the `%%crane_llm` magic, and [`ui.py`](./crane_llm/nb_extension/ui.py), its output, which run in the kernel and need no frontend
- The built-in checks: [`checker/`](./crane_llm/nb_extension/checker), with the walk through a cell in [`walker.py`](./crane_llm/nb_extension/checker/walker.py) and the rules, one module per library, in [`rules/`](./crane_llm/nb_extension/checker/rules)
- The variable tracker, which finds the cells a crash comes from: [`provenance.py`](./crane_llm/nb_extension/provenance.py)
- The check before each cell runs: [`guard.py`](./crane_llm/nb_extension/guard.py)
- The LLM prediction, as in the [CRANE-LLM paper](./PAPER.md): [`assistant.py`](./crane_llm/nb_extension/assistant.py), which asks the built-in checks first and the model next, with [`prompt_builder.py`](./crane_llm/nb_extension/prompt_builder.py) and [`runinfo.py`](./crane_llm/nb_extension/runinfo.py) sharing the runtime information extraction of [`runinfo_parser/`](./crane_llm/runinfo_parser)

Around it:

- [`eval_builtin_checks.py`](./eval_builtin_checks.py): the evaluation below
- [`results/builtin_checks/`](./results/builtin_checks): its results
- [`tests/test_nb_extension.py`](./tests/test_nb_extension.py): the extension's tests, including that every crash a check reports really happens
- [`cranellm_demo/`](./cranellm_demo): the demo notebook and its Docker image, with [its own README](./cranellm_demo/README.md)

## Evaluating the built-in checks

[`eval_builtin_checks.py`](./eval_builtin_checks.py) evaluates the built-in checks and the check before each cell runs, without calling any LLM. It replays each notebook in execution order and, before every cell, runs the checker on it with the live kernel state, as the guard does. Then it runs the cell and records whether it raised. 

On Junobench:

Run it from the repository root in the [Docker environment](./PAPER.md#environment):

```bash
python eval_builtin_checks.py                                   # all of JunoBench
python eval_builtin_checks.py --libs pandas --cases pandas_1    # one case, to try it
```

On another dataset:
```bash
python eval_builtin_checks.py --nb-root target_nbs_notebookerrorsdataset   # the second dataset
```

Another dataset must have the same layout as `target_nbs/`: `<lib>/<case>/<case>_reproduced.ipynb`, with the crash saved as the error output of the crashing cell, and optionally `<case>_fixed.ipynb`, where the same cell runs without error. Each notebook runs in its own folder, so its data files go beside it. A notebook that crashes in several places may be reproduced once per crash, as `<case>_reproduced_1.ipynb`, `<case>_reproduced_2.ipynb` and so on, as in `target_nbs_notebookerrorsdataset/`; only `_1`, the first crash, is evaluated. That dataset has no fixed notebooks, so its target cells give no precision; the earlier cells still do. 

This evaluation uses another dataset constructed from a third-party source, the [JetBrains Research Jupyter Errors dataset](https://huggingface.co/datasets/JetBrains-Research/jupyter-errors-dataset), which collects Jupyter notebooks from **GitHub** whose saved outputs contain errors. We sampled 100 of these notebooks and tried to reproduce each crash by hand, and 29 reproduced. For each one we keep the original notebook alongside one or more reproduced versions, each containing exactly one crash. Unlike JunoBench, this dataset has no diagnosis labels and no fixed versions. All notebooks run in the same Docker environment as CRANE-LLM, with a few packages the Docker image lacking:

```bash
pip install evaluate==0.4.6 intake==2.0.9 xgcm==0.9.0
pip install --no-deps scikeras==0.13.0    # its dependencies would upgrade scikit-learn
pip install intake-esm==2025.7.9          # upgrades xarray, dask, distributed, fsspec and attrs
```

One notebook (case 25) also downloads a public CMIP6 catalogue, so the replay needs network access.

The LLM pipeline of the [CRANE-LLM paper](./PAPER.md) runs on another dataset with `python main_LLM.py --dataset <dataset folder>`, and keeps its prompts and answers apart from JunoBench's, in `llms/<dataset folder>/`. See `python main_LLM.py --help` for the model, the runtime information given and the number of runs.

### Evaluation results:

The results go to `results/builtin_checks/<dataset>/`:
- `summary.txt`: on the target cells, the crashes found, the false positives on the fixed targets, and whether the exception and line are right. On every earlier cell, the cells that would have been stopped although they ran fine, and the time the check adds. Over all checked cells, the precision: a flag on a crashing target is right, a flag on a fixed target or on an earlier cell that ran fine is not. The LLM results with and without the checker answering first, and the GPT-5 tokens and cost the checker saves, counted on the recorded prompts and answers: for JunoBench, those of the CRANE-LLM paper; for another dataset, those `main_LLM.py` recorded on it, a crash being predicted when most runs say so. Run the summary again with `--summarize-only` once the LLM has answered.
- `target_cells.csv` and `prior_cells.csv`: one row per cell.
- `cells.jsonl` and `notebooks.jsonl`: the raw records. Notebooks already in them are skipped, so an interrupted run resumes where it stopped. With `--rerun` they are replayed anyway, and the new results replace the old.
- `logs/`: what each notebook printed while it was replayed.
