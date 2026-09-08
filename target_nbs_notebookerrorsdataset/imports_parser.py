import os
import json
import re
import sys
import pandas as pd

top_lib_names = [
    "pandas",
    "numpy",
    "matplotlib",
    "sklearn",
    "seaborn",
    "tensorflow",
    "torch",
    "xgboost",
    "scipy",
    "plotly",
    "cv2",
    "keras",
    "lightgbm",
    "torchvision",
    "nltk",
    "transformers",
    "catboost",
    "statsmodels",
    "imblearn",
    "wordcloud",
    "missingno",
    "optuna",
    "skimage",
    "datasets",
]

# This is simply parse and match each line of code
def get_imports_nbs_static(path_tar, get_imports_func):
    res = []
    n_total = 0
    for path, subdirs, files in os.walk(path_tar):
        for f in files:
            if f.endswith(".ipynb"):
                n_total += 1
                try:
                    with open(f"{path}/{f}", "r", encoding="utf-8") as data_file:
                        try:
                            j = json.load(data_file)
                            imports = list()
                            if ("nbformat" in j) and j["nbformat"] >=4:
                                for i,cell in enumerate(j["cells"]):
                                    if cell["cell_type"] == "code":
                                        if isinstance(cell["source"], list):
                                            for line in cell["source"]:
                                                imp = get_imports_func(line)
                                                if len(imp) > 0:
                                                    imports.extend(imp)
                                        else:
                                            for line in cell["source"].split("\n"):
                                                imp = get_imports_func(line)
                                                if len(imp) > 0:
                                                    imports.extend(imp)
                                imports = set(imports)
                                # for data_analysis_[additional-manual_labels_libraries]
                                if "keras" in imports:
                                    imports.add("tensorflow/keras")
                                    imports.remove("keras")
                                if "tensorflow" in imports:
                                    imports.add("tensorflow/keras")
                                    imports.remove("tensorflow")
                                # end
                                res.append({"fname":f, "imports":imports})
                            else:
                                print("wrong format of jupyter notebook", f)
                        except json.decoder.JSONDecodeError:
                            print("decoding error: {}".format(f))
                        except Exception as err:
                            print(f"Unexpected error converting to json {f}")
                            
                except FileNotFoundError:
                    print(f"File {f} not found.  Aborting")
                    sys.exit(1)
                except OSError:
                    print(f"OS error occurred trying to open {f}")
                    sys.exit(1)
                except Exception as err:
                    print(f"Unexpected error opening {f}")
                    sys.exit(1)
    print("Successfully parsed {0}/{1} notebook files, failed {2} ones.".format(len(res), n_total, n_total-len(res)))
    return res

# from xxx import xxx as xxx
def get_imports_line_all(line):
    line = line.strip()
    pattern = re.compile(r"(?m)^(?:from[ ]+(\S+)[ ]+)?import[ ]+(\S+)(?:[ ]+as[ ]+(\S+))?[ ]*")
    pattern_next = re.compile(r"(?m)^(\S+)(?:[ ]+as[ ]+(\S+))?[ ]*")
    imp_res = []
    imps = re.findall(pattern, line) # if it is an import statement
    if len(imps)>0:
        lines = line.split('#')[0].split(';')[0].split(",")
        if len(lines)>0: # more than 1 imports
            imps_first = re.findall(pattern, lines[0].strip()) # get the first match
            imp_res.extend(imps_first)
            imps_next_from=[imps_first[0][0]]
            for i in range(1,len(lines),1):
                #print(lines[i])
                imps_next = re.findall(pattern_next, lines[i].strip())
                #print(imps_next)
                imp_res.extend([tuple(imps_next_from+list(imp_next)) for imp_next in imps_next])
    return imp_res

# imported library name corresponding to alias in the code
# will be import names if no alias has been defined
def get_lib_alias(imps):
    res = []
    for imp in imps:
        res_item = []
        if len(imp[0])>0:
            res_item.append(imp[0].split(".")[0])
        else:
            res_item.append(imp[1].split(".")[0])
        res_item.append(imp[2]) if len(imp[2])>0 else res_item.append(imp[1])
        res.append(res_item)
    return res

def simple_lib_parser(libs_tar):
    """Returns the underlying library of the provided path."""
    if pd.isna(libs_tar):
        return None
    lib_tar = libs_tar.lower().split(",")[-1]
    if 'pandas' in lib_tar:
        return 'pandas'
    if 'torch' in lib_tar:
        return 'torch'
    if ('tensorflow' in lib_tar) or ('keras' in lib_tar):
        return 'tensorflow'
    if 'sklearn' in lib_tar:
        return 'sklearn'
    if 'matplotlib' in lib_tar:
        return 'matplotlib'
    if 'numpy' in lib_tar:
        return 'numpy'
    return lib_tar
    
def lib_alias_isML(lib_alias):
    if lib_alias:
        try:
            if isinstance(lib_alias, str):
                lib_alias = eval(lib_alias)
            elif isinstance(lib_alias, list):
                lib_alias = lib_alias
            for imp in lib_alias:
                
                if imp[0] and (simple_lib_parser(imp[0]) in top_lib_names):
                    return True
        except:
            return False
    return False