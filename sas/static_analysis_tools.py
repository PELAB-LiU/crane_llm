# static_analyzer.py
import subprocess
import tempfile
import os
from typing import List, Dict
from pathlib import Path
import json

CRASH_PRONE_PYLINT_MSG = {
    "Module 'tensorflow' has no 'keras' member",
    "Module 'tensorflow._api.v2.distribute.cluster_resolver' has no 'TCPClusterResolver' member",
    "Module 'cv2' has no 'resize' member",
    "Module 'cv2' has no 'COLOR_BGR2GRAY' member",
    "Module 'cv2' has no 'COLOR_BGR2RGB' member",
    "Module 'cv2' has no 'IMREAD_COLOR' member",
    "Module 'cv2' has no 'cvtColor' member",
    "Module 'cv2' has no 'imread' member",
    "Module 'urllib' has no 'URLopener' member",
}
CRASH_PRONE_PYRIGHT_MSG = {
    "\"keras\" is not a known attribute of module \"tensorflow\"",
    "\"keras\" is unknown import symbol",
    "Expected expression", # "6:   %tensorflow_version 2.x"
    "\"TCPClusterResolver\" is not a known attribute of module \"tensorflow._api.v2.distribute.cluster_resolver\"",
    "\"URLopener\" is not a known attribute of module \"urllib\"",
    "\"request\" is not a known attribute of module \"urllib\"",
    "\"utils\" is not a known attribute of module \"torch\"",
    "could not be determined because it refers to itself",
    "Argument missing for parameter \"self\""
}
CRASH_PRONE_PYLINT_SYMBOLS = {
    "no-name-in-module",
    "import-error",
    "syntax-error" # "%tensorflow_version 2.x"
}
CRASH_PRONE_PYRIGHT_RULES = {
    "reportMissingImports",
    "reportPrivateImportUsage",
    "reportAssignmentType",
    "reportPossiblyUnboundVariable"
}

def run_pylint(code: str) -> List[Dict]:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(["pylint", tmp_path, "--output-format=json"], capture_output=True, text=True, check=False)
        
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError:
            output = []

        lines = code.splitlines()
        diagnostics = []
        for item in output:
            lineno = item.get("line", 0)
            diagnostics.append({
                # "line": lineno,
                "code_line": f"{lineno}: {(lines[lineno - 1] if 0 < lineno <= len(lines) else '')}",
                "message": item.get("message"),
                "symbol": item.get("symbol"), # message-id
                "type": item.get("type"),
                # "file": item.get("path"),
                "source": "pylint"
            })
        return filter_pylint_crash_bugs(diagnostics)
    finally:
        os.remove(tmp_path)

def filter_pylint_crash_bugs(results: List[Dict]) -> List[Dict]:
    return [r for r in results if (r.get("type") == "error") and (r.get("symbol") not in CRASH_PRONE_PYLINT_SYMBOLS) and (r.get("message") not in CRASH_PRONE_PYLINT_MSG)]
    # return [r for r in results if r.get("type") == "error"]


def run_pyright(code: str) -> List[Dict]:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "temp.py")
        with open(file_path, "w") as f:
            f.write(code)

        # subprocess.run(["pyright", "--project", tmpdir, "--createstub", "temp"], capture_output=True)
        result = subprocess.run(["pyright", file_path, "--outputjson"], capture_output=True, text=True)

        try:
            output = json.loads(result.stdout)
            diagnostics = []
            lines = code.splitlines()

            for diag in output.get("generalDiagnostics", []):
                lineno = diag["range"]["start"]["line"]
                if "No cell has been executed" in lines[lineno]:
                    continue
                diagnostics.append({
                    # "line": lineno + 1,
                    "code_line": f"{lineno + 1}: {(lines[lineno] if 0 <= lineno < len(lines) else '')}",
                    "message": diag["message"],
                    "rule": diag.get("rule"),
                    # "file": diag.get("file"),
                    "severity": diag.get("severity"),
                    "source": "pyright"
                })

            return filter_pyright_crash_bugs(diagnostics)
        except json.JSONDecodeError:
            return []

from typing import List, Dict

def filter_pyright_crash_bugs(results: List[Dict]) -> List[Dict]:
    filtered = []
    for r in results:
        severity = r.get("severity", "")
        rule = r.get("rule", "")
        message = r.get("message", "")
        
        # Skip messages that match any known non-crash pattern
        skip = False
        for pattern in CRASH_PRONE_PYRIGHT_MSG:
            if pattern in message:
                skip = True
                break

        if severity == "error" and (rule not in CRASH_PRONE_PYRIGHT_RULES) and (not skip):
            filtered.append(r)
    return filtered

def run_sas_single(path: Path, sa_name: str):
    if path.suffix == ".txt":
        code = path.read_text(encoding='utf-8')
        if sa_name == "pylint":
            results = run_pylint(code)
        elif sa_name == "pyright":
            results = run_pyright(code)
        else:
            results = f"not support sa tool: {sa_name}."
        return results
    else:
        print('not supported file type.')

def run_sas_all(sa_name: str, input_path: Path, output_path: Path, case_names: List[str] = None):
    os.makedirs(output_path, exist_ok=True)
    for filename in os.listdir(input_path):
        output_name = os.path.splitext(filename)[0]
        if case_names and ((output_name.split("_")[0]+"_"+output_name.split("_")[1]) not in case_names):
            continue
        output_file = f"{output_path}/{output_name}.json"

        print(f"Predicting {output_name} with {sa_name}...")
        res = run_sas_single(Path(os.path.join(input_path, filename)), sa_name)

        with open(output_file, "w") as f:
            json.dump(res, f, indent=2)
            print(f"Results are saved in {output_file}.")