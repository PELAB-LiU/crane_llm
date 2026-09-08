import os
import json
from pathlib import Path
import pandas as pd
from slugify import slugify
import imports_parser
import error_parser
import shutil

nbpath = 'C:/Users/yirwa29/Downloads/Dataset-Nb/Docker_kaggle_env/jupyter-errors-dataset/parsed_nbs'
datapath = 'C:/Users/yirwa29/Downloads/Dataset-Nb/Docker_kaggle_env/jupyter-errors-dataset/data'

nameprocessed = 'parsed_summary_without_content_processed.xlsx'

ename_include = [
    "NameError",
    "TypeError",
    "ValueError",
    "AttributeError",
    "KeyError",
    "IndexError",
    "RuntimeError",
    "Exception",
    "ZeroDivisionError",
    "UnboundLocalError",
    "StopIteration",
    "InvalidArgumentError",
    "ValidationError",
    "error",
    "LinAlgError",
    "CoaKeyError",
    "DuplicateFlagError",
    "AnalysisException",
]

def make_fname(row):
    p = Path(row["path"])
    return slugify(f"{row['id']}_{row['repo_name']}_{p.stem}") + p.suffix

def parse_notebooks():
    n_decoding_error = 0

    full_df = pd.DataFrame()
    for i in range(17):
        id_name = str(i).zfill(5)
        filepath = os.path.join(datapath, f'train-{id_name}-of-00017.parquet')
        savepath = os.path.join(nbpath, id_name)
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        if os.path.isfile(filepath):
            df = pd.read_parquet(filepath)
            df["fname"] = df.apply(make_fname, axis=1)
            full_df = pd.concat([full_df, df.drop(columns=['content'])], ignore_index=True)

            for _, row in df.iterrows():
                c = row['content']
                file_name = row['fname']
                try:
                    file_as_json = json.loads(c)
                    with open(os.path.join(savepath, file_name), 'w', encoding='utf-8') as f:
                        json.dump(file_as_json, f, ensure_ascii=False, indent=4)
                except json.decoder.JSONDecodeError:
                    n_decoding_error += 1
                    print(f"decoding error: {file_name}")
                
    print(f"\n total number of notebooks that cannot be decoded: {n_decoding_error}") # 20

    full_df = full_df.sort_values("id")
    full_df.to_excel(os.path.join(nbpath, 'parsed_summary_without_content.xlsx'), index=False)

    print(f"Total notebooks: {len(full_df)}, Unique fnames: {full_df['fname'].nunique()}, Unique ids: {full_df['id'].nunique()}") # 10000 10000 6109
    print(f"Duplicate ids: {full_df['id'].duplicated().sum()}") # 3891
    print(f"Duplicate fnames: {full_df['fname'].duplicated().sum()}") # 0

def filter_notebooks():
    full_df = pd.read_excel(os.path.join(nbpath, 'parsed_summary_without_content.xlsx'))

    # imports
    res = imports_parser.get_imports_nbs_static(nbpath, imports_parser.get_imports_line_all)
    res_pd = pd.DataFrame.from_dict(res)
    res_pd["lib_alias"] = res_pd.imports.apply(imports_parser.get_lib_alias)
    res_pd["is_MLnb"] = res_pd.lib_alias.apply(imports_parser.lib_alias_isML)
    full_df = full_df.merge(res_pd, on="fname", how="left")
    # Successfully parsed 9954/9976 notebook files, failed 22 ones.

    # errors
    res_errs = error_parser.filter_notebooks_with_errors(nbpath, is_resave = False, path_des = None)
    res_errs["traceback"] = res_errs["traceback"].map(
        lambda tb: error_parser.parse_traceback("\n".join(tb))
    )
    full_df = full_df.merge(res_errs, on="fname", how="left")
    # Total number of notebooks containing error: 9916
    # Total number of notebooks that cannot be decoded: 22

    full_df.to_excel(os.path.join(nbpath, nameprocessed), index=False)
    full_df["ename"].value_counts().to_csv(os.path.join(nbpath, "ename_counts.csv"))
    # compile a list of included enames manually, and then filter the dataframe based on that list in the next step

def sample_notebooks(sample_size=100, sample_round=1):
    full_df = pd.read_excel(os.path.join(nbpath, nameprocessed))
    
    # sample notebooks from the pool
    df_pool = full_df[(full_df["if_ename_include"] == True) & (full_df["is_MLnb"]==1)]
    df_pool = (df_pool.drop_duplicates(subset="fname").sample(n=sample_size, random_state=42))
    full_df.loc[full_df["fname"].isin(df_pool["fname"]), "sampled_round"] = sample_round
    full_df.to_excel(os.path.join(nbpath, nameprocessed), index=False)

    # get the corresponding notebook files and save them to a new folder
    sample_path = os.path.join(nbpath, 'sampled_nbs', f'round_{sample_round}')
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    copied_files = 0
    for path, subdirs, files in os.walk(nbpath):
        if os.path.abspath(path).startswith(os.path.abspath(sample_path)):
            continue
        for f in files:
            if f.endswith(".ipynb") and f in full_df[full_df["sampled_round"]==sample_round]["fname"].values:
                shutil.copy2(os.path.join(path, f), os.path.join(sample_path, f))
                copied_files += 1
    print(f"Sampled {sample_size} notebooks from {len(df_pool)} notebook pool. Number of copied files: {copied_files}.")

def main():
    # parse_notebooks()
    # filter_notebooks()

    # full_df = pd.read_excel(os.path.join(nbpath, nameprocessed))
    # full_df["if_ename_include"] = full_df["ename"].apply(lambda ename: ename in ename_include)
    # full_df.to_excel(os.path.join(nbpath, nameprocessed), index=False)

    sample_notebooks(sample_size=100, sample_round=1)
    # pass

if __name__ == "__main__":
    main()