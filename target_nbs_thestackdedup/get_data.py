from pathlib import Path
import re
import unicodedata

import pandas as pd
from datasets import load_dataset

label_if_ML_bug = {"ML bug": ["ML/data science library related (ML imports, error raised by library)"],
                  "python bug": ["general code error"]}

label_refined_exp_type = {"variable not found": ["variable not found"],  # name error
                          "invalid argument": ["wrong arguments to API"],
                          "module not found": ["module not found"],  # name error
                          "attribute error": ["attributeerror"],
                          "key error": ["keyerror", "notfounderror"],
                          "tensor shape mismatch": ["tensor shape mismatch"],  # value error
                          "data value violation": ["valueerror - data value violation"],  # value error
                          "name error": ["function not found ", "class not found", "nameerror"],  # name error
                          "value error": ["cast exception", "valueerror - data range mismatch", "valueerror"],  # value error
                          "index error": ["indexerror-nd", "indexerror-1d"],
                          "OOM": ["out of memory (OOM)"],
                          "type error": ["typeerror", "typeerror-notcallable", "typeerror-op", "typeerror-notsubscriptable", "typeerror-notiterable", "typeerror-unhashable"],
                          "request error": ["requesterror"],
                          "unsupported broadcast": ["unsupported broadcast"],  # value error
                          "runtime error": ["runtimeerror"],
                          "model initialization error": ["initialization error (call mul-times, wrong order)"],
                          "environment error": ["importerror", "environment setup"],
                          "feature name mismatch": ["valueerror - feature name mismatch"],  # value error
                          "other": ["syntaxerror", "indentationerror", "zerodivisionerror", "assertionerror", "systemerror", "executablenotfound", "out of space (disk)", "unknown"],
                          "io error": ["filenotfounderror", "unsupported file type (read file)", "file permission", "fileexistserror", "jsondecodeerror", "incompleteparseerror"]}

source_xlsx_path = Path("C:/Users/yirwa29/Downloads/data_jupyter_nbs_empirical/Manual_labeling/cluster_sampled_labeled_final.xlsx")
output_path = Path(__file__).with_name("cluster_sampled_labeled_final_filtered.xlsx")
ds_meta_cache_path = Path(__file__).with_name("stack_jupyter_meta.pkl")


def make_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-")

if ds_meta_cache_path.exists():
    ds_meta = pd.read_pickle(ds_meta_cache_path)
else:
    ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/jupyter-notebook", split="train")
    ds_meta = ds.select_columns(["max_stars_repo_name", "max_stars_repo_head_hexsha", "max_stars_repo_path"]).to_pandas()
    ds_meta.to_pickle(ds_meta_cache_path)

df_mlerr_labeled = pd.read_excel(source_xlsx_path, keep_default_na=False)
df_mlerr_labeled_filtered = df_mlerr_labeled[df_mlerr_labeled.nb_source == 2].copy()
df_mlerr_labeled_filtered = df_mlerr_labeled_filtered[df_mlerr_labeled_filtered.label_if_ML_bug.isin(label_if_ML_bug["ML bug"])].copy()
df_mlerr_labeled_filtered = df_mlerr_labeled_filtered[
    df_mlerr_labeled_filtered.label_refined_exp_type.isin(
        label_refined_exp_type["module not found"]
        + label_refined_exp_type["OOM"]
        + label_refined_exp_type["request error"]
        + label_refined_exp_type["runtime error"]
        + label_refined_exp_type["environment error"]
        + label_refined_exp_type["other"]
        + label_refined_exp_type["io error"]
    )
].copy()

if "fname" not in df_mlerr_labeled_filtered.columns:
    raise KeyError("Expected column 'fname' in the labeled spreadsheet.")

ds_meta = ds_meta.copy()
ds_meta["nb_key"] = ds_meta["max_stars_repo_path"].astype(str).map(lambda path: make_slug(Path(path).name.rsplit(".", 1)[0]))
df_mlerr_labeled_filtered["nb_key"] = df_mlerr_labeled_filtered["fname"].astype(str).map(
    lambda name: make_slug("-".join(str(name).rsplit(".", 1)[0].split("-")[2:]))
)
print(df_mlerr_labeled_filtered[["fname", "nb_key"]].head(10))
print(ds_meta[["nb_key"]].head(10))
df_mlerr_labeled_filtered = df_mlerr_labeled_filtered.merge(
    ds_meta,
    on="nb_key",
    how="left",
)

df_mlerr_labeled_filtered["github_link"] = (
    "https://github.com/"
    + df_mlerr_labeled_filtered["max_stars_repo_name"].astype(str)
    + "/blob/"
    + df_mlerr_labeled_filtered["max_stars_repo_head_hexsha"].astype(str)
    + "/"
    + df_mlerr_labeled_filtered["max_stars_repo_path"].astype(str).str.replace(" ", "%20", regex=False)
)

df_mlerr_labeled_filtered.drop(columns=["nb_url_kaggle"], errors="ignore").to_excel(output_path, index=False)
