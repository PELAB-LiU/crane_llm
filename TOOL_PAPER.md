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

- [`eval_builtin_checks.py`](./eval_builtin_checks.py): the evaluation of the built-in checks below
- [`results/builtin_checks/`](./results/builtin_checks): its results
- [`eval_tracer.py`](./eval_tracer.py): the evaluation of the variable tracker below, with its results in `results/tracer/`
- [`tests/test_nb_extension.py`](./tests/test_nb_extension.py): the extension's tests, including that every crash a check reports really happens
- [`cranellm_demo/`](./cranellm_demo): the demo notebook and its Docker image, with [its own README](./cranellm_demo/README.md)

## The 48 built-in checks

The checker detects 48 kinds of crash. Each has an id, which the evaluation records as `finding_rule` and counts under *Crashes found by rule* in `summary.txt`. The [user guide's table](./README_PYPI.md#built-in-checks) describes the same checks for users, grouped by what the code does, so its rows do not correspond one to one to the ids below.

Six are found by the walk through the cell itself, in [`walker.py`](./crane_llm/nb_extension/checker/walker.py), while it looks up names, attributes and modules:

| Id | Raises | Example |
|---|---|---|
| `undefined-name` | `NameError` | `print(data1)` where no `data1` was ever defined |
| `missing-attribute` | `AttributeError` | `df.head()` after `df = df.dropna(inplace=True)` left `df` as `None`; `df.append(row)` on pandas 2 |
| `missing-module` | `ModuleNotFoundError`, `ImportError` | `from sklearn import cross_validation` |
| `unpack-count` | `ValueError` | `a, b = (1, 2, 3)` |
| `explicit-raise` | the raised exception, `TypeError` | `raise ValueError("bad input")` outside any `try` |
| `syntax` | `SyntaxError`, `IndentationError` | `print "hello"` |

The other 42 are rules in [`rules/`](./crane_llm/nb_extension/checker/rules). Some ids are checked for several libraries; each is listed once, with every library it covers:

| Id | Libraries | Raises | Example |
|---|---|---|---|
| `missing-key` | Python | `KeyError` | `d["x"]` or `del d["x"]` for a key the dict does not have; `os.environ["API_TOKEN"]` when it is not set |
| `unhashable-key` | Python | `TypeError` | `{[1, 2]: "a"}`, `[1, 2] in some_dict` |
| `index-range` | Python, pandas, NumPy | `IndexError` | `lst[5]` on a list of 3; `df.iloc[100]` on 10 rows; `a[3]` on an array of shape `(3,)` |
| `index-type` | Python, NumPy | `TypeError`, `ValueError`, `IndexError` | `lst["a"]`, `lst[::0]`, `arr["price"]` on a plain array |
| `not-subscriptable` | Python | `TypeError` | `x[0]` where `x` is `None` |
| `immutable-assignment` | Python | `TypeError` | `t[0] = 1` on a tuple; `del s[0]` on a string |
| `operand-types` | Python | `TypeError` | `None + 1` |
| `division-by-zero` | Python | `ZeroDivisionError` | `x / 0` |
| `bad-format` | Python | `IndexError`, `KeyError`, `ValueError`, `TypeError` | `"{} {}".format(x)`, `"%d %d" % (x,)`, `f"{1.5:d}"` |
| `bad-operand` | Python | `TypeError` | `-"a"`, `-None` |
| `unsupported-comparison` | Python | `TypeError` | `x in 5`, `None > 0` |
| `not-callable` | Python | `TypeError` | `df.shape()` |
| `bad-argument` | Python, scikit-learn | `TypeError`, `ValueError` | `pd.read_csv(path, error_bad_lines=False)`; `LogisticRegression(n_estimators=3)`; `train_test_split(X, test_size=1.5)` |
| `abstract-class` | Python | `TypeError` | instantiating a class that still has abstract methods |
| `len-unsized` | Python | `TypeError` | `len(None)` |
| `conversion` | Python | `ValueError`, `OverflowError` | `int("3.5")` |
| `range-step` | Python | `ValueError` | `range(0, 10, 0)` |
| `not-iterable` | Python | `TypeError` | `for x in None`; `f(*5)` |
| `not-a-mapping` | Python | `TypeError` | `f(**None)` |
| `missing-file` | Python | `FileNotFoundError` | `pd.read_csv("data/train.csv")` when there is no such file |
| `missing-column` | pandas | `KeyError` | `df["col"]`, `df.drop(columns=["col"])`, `df.groupby("col")`, `pd.merge(a, b, on="id")` without that column |
| `column-length` | pandas | `ValueError` | `df["col"] = [1, 2]` on a frame of 10 rows |
| `missing-label` | pandas | `KeyError` | `df.loc["row"]` without that label |
| `astype-int` | pandas, NumPy | `ValueError`, `IntCastingNaNError`, `TypeError` | `df["id"].astype(int)` on a column holding `"n/a"` or NaN |
| `length-mismatch` | pandas | `ValueError` | `s + [1, 2]` on a Series of 3 |
| `ambiguous-truth` | pandas, NumPy | `ValueError` | `if df:`, `not arr` |
| `empty-concat` | pandas | `ValueError` | `pd.concat([])` |
| `broadcast` | NumPy, PyTorch | `ValueError`, `RuntimeError` | `a + b` with shapes `(3,)` and `(4,)` |
| `matmul-shape` | NumPy, PyTorch | `ValueError`, `RuntimeError` | `a @ b` with shapes `(2, 3)` and `(2, 3)`; an `nn.Linear(10, 1)` layer called on input whose last dimension is 5 |
| `concatenate-shape` | NumPy, PyTorch | `ValueError`, `RuntimeError` | `np.vstack([a, b])`, `torch.cat([a, b])` of pieces that do not fit |
| `reshape-size` | NumPy, PyTorch | `ValueError`, `RuntimeError` | `a.reshape(3, 4)` on 10 elements |
| `axis-range` | NumPy | `AxisError` | `a.sum(axis=3)` on a 2-D array |
| `array-dimensions` | NumPy, matplotlib | `ValueError`, `TypeError` | `np.savetxt(path, a)` with a 3-D `a`; `plt.imshow(a)` with a 1-D `a` |
| `non-finite-data` | NumPy, scikit-learn | `ValueError` | `np.histogram(a)` or `model.fit(X, y)` where the data holds NaN or infinity |
| `not-fitted` | scikit-learn | `NotFittedError` | `model.predict(X)` before `model.fit` |
| `feature-names` | scikit-learn | `ValueError` | `model.predict(df)` with other columns, or the same in another order, than at fit time |
| `feature-count` | scikit-learn | `ValueError` | `model.predict(X)` with 5 columns for a model fitted on 6 |
| `sample-count` | scikit-learn | `ValueError` | `model.fit(X, y)` or `train_test_split(X, y)` with `X` and `y` of different lengths |
| `too-few-samples` | scikit-learn | `ValueError` | `knn.predict(X)` with `n_neighbors` larger than the number of samples fitted |
| `continuous-target` | scikit-learn | `ValueError` | `LogisticRegression().fit(X, y)` or `accuracy_score(y, scores)` with continuous values |
| `bad-value` | scikit-learn | `ValueError` | `model.fit(df, y)` where a column holds text such as `"male"` |
| `tensor-conversion` | PyTorch | `RuntimeError` | `t.numpy()` on a tensor that requires grad; `t.item()` on a tensor of several values |

There are no Keras rules. The general checks still apply to Keras objects: a metric missing from `history.history`, which is a plain dict, is a `missing-key`, and a method a model does not have is a `missing-attribute`.

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

## Evaluating the variable tracker

[`eval_tracer.py`](./eval_tracer.py) checks whether the variable tracker points to the cell that was fixed. JunoBench has no labels for the cells a crash comes from, so the fixed cells are taken from the diff of each reproduced notebook and its fixed version. A crash is scored when the fix changed a cell other than the crashing one: a cell that ran before the crash, or a cell that never ran and assigns a name the crashing cell reads, provided it comes before the crashing cell or the fix says to execute it (`# fix --- execute this cell beforehand`). Results are also given separately for the `NBspecific_*` cases. Crashes whose fix changes only the crashing cell are counted but not scored, since the verdict already points to that cell.

Each reproduced notebook is replayed in execution order with the tracker recording, as it does in the kernel. At the crashing cell, the tracker is asked about three sets of blamed variables:
- `fix`: the names the crashing cell reads that the fix touches, in a changed line of code or as a name a changed cell assigns. JunoBench marks fixes that only change which cells run with comments such as `# fix --- execute this cell`, so for those the assigned names are what counts.
- `read`: every name the crashing cell reads, except imported modules.
- `checker`: the variables the built-in checker blamed, where it found the crash.
- `llm_<model>`, for GPT-5, Gemini and Qwen: the variables the LLM blamed, where it predicted the crash. The answers are those the CRANE-LLM paper recorded with runtime information, in `llms/llms_outputs/results_raw/crash_detection_<model>_runinfo/`; a crash is predicted when most runs say so. These answers have no `variables` field, so the variables are found as the extension finds them when a model names none: the names the crashing cell reads that the reasoning of the first run predicting a crash mentions. No LLM is called.
- `checker+llm_<model>`: the checker's variables where it found the crash, and the LLM's otherwise, as the extension does.

A crash counts as found when the reported cells include at least one fixed cell. The number of cells reported is given beside it, since reporting every earlier cell would always find it.

Run it from the repository root in the [Docker environment](./PAPER.md#environment). It needs fixed notebooks, so it runs on JunoBench only:

```bash
python eval_tracer.py --nb-root target_nbs_junobench
python eval_tracer.py --nb-root target_nbs_junobench --libs pandas --cases pandas_1   # one case, to try it
```

The results go to `results/tracer/<dataset>/`:
- `summary.txt`: what the fixes changed, and for each set of variables, how often a fixed cell is reported, whether through the record of executed cells or the search for cells that would define a missing variable, and how many cells are reported.
- `cases.csv` and `scored_cases.csv`: one row per crash, with the fixed cells, the variables and the cells reported.
- `cases.jsonl` and `notebooks.jsonl`: the raw records. As with the built-in checks, finished notebooks are skipped when the script runs again, unless `--rerun` is given.
- `logs/`: what each notebook printed while it was replayed.

The replay follows the execution counts saved in each notebook, so it reproduces out-of-order execution but not reruns: a notebook keeps only the last execution count of each cell.
