import os
import json
import shutil
import re
from uuid import uuid3, NAMESPACE_DNS, UUID
from pathlib import Path
import pandas as pd

def parse_traceback(str_traceback):
    """Parses the traceback to remove all ansii escape characters."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    result = ansi_escape.sub('', str_traceback)
    return result

def generate_uuid_for_nb_exception(
    file_name: Path | str, nb_cell_index: int, exception_name: str
) -> UUID:
    """
    Generates a unique ID of an exception, using `uuid3` and the DNS
    namespace. This process is not case-sensitive.
    :param file_name: the path in which the exception was thrown.
    This can be any type of path, only the file name and file extension
    are used.
    :param cell_index: The cell in which the exception was raised. This
    is the cell index in the note book (i.e., counting markdown cells
    etc. as well). This is zero-indexed.
    :param exception_name: The type of the exception.
    """

    if isinstance(file_name, str):
        file_name = Path(file_name)

    file_name = file_name.name
    nb_cell_index = str(nb_cell_index)

    entry = f"nb_exception://{file_name}.{nb_cell_index}.{exception_name}"
    entry = entry.lower()

    entry_id = uuid3(NAMESPACE_DNS, entry)
    return entry_id

def is_contain_error_output(file_name, file_as_json):
    res = 0
    res_err = []
    for cell_index, cell in enumerate(file_as_json["cells"]):
        if cell["cell_type"] == "code" and "outputs" in cell:
            for output in cell["outputs"]:
                if output["output_type"]=="error":
#                     if output["ename"] in ['FileNotFoundError', 'KeyboardInterrupt']: #ignore error types
#                         continue
#                     print(output["ename"])
                    res = 1
                    exception_id = generate_uuid_for_nb_exception(
                        file_name, cell_index, output["ename"])
                    res_err.append((file_name, exception_id, output["ename"], output["evalue"], output["traceback"]))
    return res, res_err

def filter_notebooks_with_errors(path_tar, is_resave = False, path_des = None):
    total_notebook = 0
#     total_notebook2 = 0
    cell_index = 0
    n_decoding_error = 0
    res_errs = []
    print("\nStarted filtering:")
    for path, subdirs, files in os.walk(path_tar):
        for f in files:
            if f.endswith(".ipynb"):
                try:
                    file = open(f"{path}/{f}", "r", encoding="utf-8")
                    try:
                        file_as_json = json.loads(file.read())
                        if not file_as_json.get("cells"):
                            raise Exception('No cells property in the notebook, probably a very old version')
                        res, res_err = is_contain_error_output(f, file_as_json)
                        if res == 1:
                            total_notebook += 1
                            if is_resave: shutil.copyfile(f"{path}/{f}", f"{path_des}/{f}")
                        if res > 0:
                            res_errs.extend(res_err)

                    except json.decoder.JSONDecodeError:
                        n_decoding_error += 1
                        print("decoding error: {}".format(f))
                    except Exception as error:
                        n_decoding_error += 1
                except Exception:
                    print("Errors occur when opening file {}".format(f))
                    n_decoding_error += 1
    print("\nTotal number of notebooks containing error: {}".format(total_notebook))
#     print("Total number of notebooks containing ValueError: {}".format(total_notebook2))
    print("Total number of notebooks that cannot be decoded: {}".format(n_decoding_error))
    return pd.DataFrame(res_errs, columns=['fname', 'eid', 'ename', 'evalue', 'traceback'])
