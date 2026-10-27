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

[`eval_builtin_checks.py`](./eval_builtin_checks.py) evaluates the built-in checks and the check before each cell runs, without calling any LLM. It replays each notebook in execution order and, before every cell, runs the checker on it with the live kernel state, as the guard does. Then it runs the cell and records whether it raised. Run it from the repository root in the [Docker environment](./PAPER.md#environment):

```bash
python eval_builtin_checks.py                                   # all of JunoBench
python eval_builtin_checks.py --libs pandas --cases pandas_1    # one case, to try it
python eval_builtin_checks.py --nb-root other_nbs               # another dataset
```

Another dataset must have the same layout as `target_nbs/`: `<lib>/<case>/<case>_reproduced.ipynb`, with the crash saved as the error output of the crashing cell, and optionally `<case>_fixed.ipynb`, where the same cell runs without error. Case folders are named `<name>_<number>`, and each notebook runs in its own folder, so its data files go beside it.

The results go to `results/builtin_checks/<dataset>/`:
- `summary.txt`: on the target cells, the crashes found, the false positives on the fixed targets, and whether the exception and line are right. On every earlier cell, the cells that would have been stopped although they ran fine, and the time the check adds. For JunoBench, the CRANE-LLM paper's LLM results with and without the checker answering first, and the GPT-5 tokens and cost the checker saves, counted on the prompts and answers that paper recorded.
- `target_cells.csv` and `prior_cells.csv`: one row per cell.
- `cells.jsonl` and `notebooks.jsonl`: the raw records. Notebooks already in them are skipped, so an interrupted run resumes where it stopped. With `--rerun` they are replayed anyway, and the new results replace the old.
- `logs/`: what each notebook printed while it was replayed.
