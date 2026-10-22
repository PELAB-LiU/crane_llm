# Demo: 

A short regression notebook with six cells to check, five of which crash, to see each part of CRANE-LLM at work: the built-in checks, the LLM prediction, the runtime information, the origin of a crash, and checking cells before they run.

The notebook is based on a real Kaggle notebook on the [Seoul Bike Trip](https://www.kaggle.com/datasets/tagg27/seoul-bike-trip) dataset (CC BY 4.0), and scene 2 is the crash that notebook actually hit. The data in `data/` is a synthetic copy of the original.

To see it before running it yourself, watch the walkthrough:

https://github.com/user-attachments/assets/7dfc82b4-2444-4448-bc6b-2af57091c988

## Running the demo

Scenes 1, 5 and 6 use only the built-in checks and need no API key. Scenes 2 to 4 ask the LLM model, so they need a key; without one, switch *Use the LLM* off and they report what the built-in checks found. Whichever way you run it, start from a fresh kernel with *Check cells before they run* switched **off**.

### With Docker

The image has JupyterLab, CRANE-LLM, the libraries the notebook uses and the notebook itself, all at fixed versions. With [Docker](https://docs.docker.com/get-docker/) installed:

```bash
docker run --rm -p 127.0.0.1:8888:8888 -e CRANE_LLM_API_KEY=sk-... ghcr.io/yarinamomo/crane-llm-demo:1.0.1
```

Then open <http://localhost:8888>, which opens the demo notebook. Leave out `-e CRANE_LLM_API_KEY=...` to use the built-in checks only, and add `-e CRANE_LLM_MODEL=...` and `-e CRANE_LLM_BASE_URL=...` for a provider other than OpenAI. Stop it with Ctrl+C; nothing is kept after it stops.

### In your own JupyterLab

- JupyterLab with CRANE-LLM installed (`pip install "crane-llm[lab]"`), plus `pandas`, `scikit-learn`, `matplotlib` and `seaborn`.
- An API key set once with `crane_llm.set_api_key(...)`.
- Open `cranellm_demo.ipynb` from this folder, so that it finds `data/`.

### On Kaggle

Use the [same demo on Kaggle](https://www.kaggle.com/code/yarinawang/crane-llm-demo). It uses the `%%crane_llm` cell magic instead of the toolbar button; click **Copy & Edit** to run it in your own Kaggle account.

## The scenes

Run the notebook from the top. The purple boxes in the notebook mark the scenes, and each box also says how to fix the crash, in small grey text.

| Scene | What CRANE-LLM shows |
|---|---|
| 1 | **Built-in check · certain**: `AttributeError`, `data0` is `None`. *Origin of the crash* points to `data0 = data.dropna(inplace=True)` |
| 2 | **LLM prediction**: crash. The built-in checks find nothing certain, and the model reads the code: `data1` is a `Categorical`, not a DataFrame |
| 3 | **LLM prediction**: no crash (green) |
| 4 | **LLM prediction**: `KeyError`, because the column is called `Humid`, not `Humidity`. With runtime information off, usually green |
| 5 | **Built-in check · certain**: `ValueError`, `X_test` has a column `RushHour` the scaler was not fitted on. *Origin* lists the cell that added it and the cell that fitted `scaler` |
| 6 | **Built-in check · certain**: `NameError` on `rf_pred`. With *Check cells before they run* on, the cell is stopped at once with `CrashPrevented` instead of failing after 30 s of training |
