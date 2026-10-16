# Demo: 

A short regression notebook with six cells to check, five of which crash, to see each part of CRANE-LLM at work: the built-in checks, the LLM prediction, the runtime information, the origin of a crash, and checking cells before they run.

The notebook is based on a real Kaggle notebook on the [Seoul Bike Trip](https://www.kaggle.com/datasets/tagg27/seoul-bike-trip) dataset (CC BY 4.0), and scene 2 is the crash that notebook actually hit. The data in `data/` is a synthetic copy of the original.

## Before you start

- JupyterLab with CRANE-LLM installed (`pip install "crane-llm[lab]"`), plus `pandas`, `scikit-learn`, `matplotlib` and `seaborn`.
- An API key set once with `crane_llm.set_api_key(...)`. Scenes 1, 5 and 6 need none; scenes 2 to 4 use the model.
- Start from a fresh kernel, with *Check cells before they run* switched **off**.

## The scenes

Run the notebook from the top. The purple boxes in the notebook mark the scenes, and each fix is commented in its cell after `# Fix`.

| Scene | What CRANE-LLM shows |
|---|---|
| 1 | **Built-in check · certain**: `AttributeError`, `data0` is `None`. *Origin of the crash* points to `data0 = data.dropna(inplace=True)` |
| 2 | **LLM prediction**: crash. The built-in checks find nothing certain, and the model reads the code: `data1` is a `Categorical`, not a DataFrame |
| 3 | **LLM prediction**: no crash (green) |
| 4 | **LLM prediction**: `KeyError`, because the column is called `Humid`, not `Humidity`. With runtime information off, usually green |
| 5 | **Built-in check · certain**: `ValueError`, `X_test` has a column `RushHour` the scaler was not fitted on. *Origin* lists the cell that added it and the cell that fitted `scaler` |
| 6 | **Built-in check · certain**: `NameError` on `rf_pred`. With *Check cells before they run* on, the cell is stopped at once with `CrashPrevented` instead of failing after 30 s of training |
