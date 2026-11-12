"""Evaluate the variable tracer: does it point to the cell that was fixed?

JunoBench gives, for each crash, the reproduced notebook and a fixed version.
Matching their code cells shows which cells the fix changed. When the fix
changes a cell other than the crashing one, that cell is where the user had
to look, and the tracer should report it.

Each reproduced notebook is replayed in execution order, as in
eval_builtin_checks.py, with the tracer recording around every cell as it
does in the kernel. At the crashing cell, the tracer is asked about the
variables blamed for the crash, chosen in three ways:

- fix: the names the crashing cell reads that the fix touches: they appear
  in a code line the fix changed, or a cell the fix changed assigns them.
  JunoBench marks a fix that only changes which cells run with a comment
  such as "# fix --- execute this cell", so for those the second part is
  what counts. The variables a correct verdict would blame, as far as the
  diff shows them; this measures the tracer alone.
- read: every name the crashing cell reads, except imported modules. Needs
  no ground truth, but reports more cells.
- checker: the variables the built-in checker blamed, where it found the
  crash. What a user gets without an LLM.
- llm_<model>: the variables the LLM blamed, where it predicted the crash.
  The answers are those the CRANE-LLM paper recorded, with runtime
  information; a crash is predicted when most runs say so. They have no
  "variables" field, so the variables are found as the extension does when a
  model names none: the names the cell reads that the reasoning of the first
  run predicting a crash mentions.
- checker+llm_<model>: the checker's variables where it found the crash, and
  otherwise the LLM's: what the extension does.

A fixed cell that ran before the crash is in the tracer's history. One that
never ran is not; the tracer then searches the notebook for cells that would
define a missing variable, and such a cell counts only if it assigns a name
the crashing cell reads and either comes before the crashing cell or the fix
says to execute it. A later cell that the fix also edited and that happens to
reassign a name, such as another plot cell assigning ``ax``, does not count.
Cases named NBspecific_* are reported separately as well. A crash is scored if the fix changed any such cell,
and the tracer is right if it reports at least one of them. Crashes whose fix
only changes the crashing cell have no earlier cell to find and are not
scored: the verdict already points to that cell.

    python eval_tracer.py --nb-root target_nbs_junobench
    python eval_tracer.py --nb-root target_nbs_junobench --libs pandas --cases pandas_1
    python eval_tracer.py --nb-root target_nbs_junobench --summarize-only
"""

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import pandas as pd

from eval_builtin_checks import LLM_MODELS, append_jsonl, list_notebooks, majority_detection, ratio, read_jsonl

SETUPS = ["fix", "read", "checker"] + [f"llm_{model}" for model in LLM_MODELS]
COMBINED = [f"checker+llm_{model}" for model in LLM_MODELS]
PATH_LLM_ANSWERS = Path(__file__).resolve().parent.joinpath("llms", "llms_outputs", "results_raw")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_EXECUTE_MARK = re.compile(r"#\s*fix\b.*\bexecute\b", re.IGNORECASE)


# --- the fix, from the diff ------------------------------------------------------


def code_sources(path_nb):
    with open(path_nb, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    sources = []
    for cell in cells:
        if cell["cell_type"] == "code":
            source = cell["source"]
            sources.append("".join(source) if isinstance(source, list) else source)
    return sources


def changed_lines(old, new):
    """The lines one version of a cell has and the other has not."""
    return [line[2:] for line in difflib.ndiff(old.splitlines(), new.splitlines()) if line[:2] in ("- ", "+ ")]


def diff_fix(path_reproduced, path_fixed):
    """The code cells of the reproduced notebook that the fix changed or deleted,
    the lines it changed anywhere, whether it inserted new cells, and the changed
    cells it marks as ones to execute ("# fix --- execute this cell beforehand")."""
    old, new = code_sources(path_reproduced), code_sources(path_fixed)
    matcher = difflib.SequenceMatcher(a=[s.strip() for s in old], b=[s.strip() for s in new], autojunk=False)
    changed, lines, inserted, marked = set(), [], False, set()
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        changed.update(range(i1, i2))
        inserted |= (j2 - j1) > (i2 - i1)
        for k in range(max(i2 - i1, j2 - j1)):
            before = old[i1 + k] if i1 + k < i2 else ""
            after = new[j1 + k] if j1 + k < j2 else ""
            cell_lines = changed_lines(before, after)
            lines += cell_lines
            if i1 + k < i2 and any(_EXECUTE_MARK.search(line) for line in cell_lines):
                marked.add(i1 + k)
    return changed, lines, inserted, marked


def counts_as_fixed_not_run(cell, target, marked):
    """A changed cell that never ran and assigns a name the crashing cell reads
    is where the fix lies only if it comes before the crashing cell, or the fix
    says to execute it. A later cell that merely reassigns the same name, such
    as another plot cell assigning ``ax``, is a separate edit."""
    return cell < target or cell in marked


def names_in(lines):
    """The identifiers in lines of code, leaving out comments."""
    return {name for line in lines for name in _IDENTIFIER.findall(line.split("#", 1)[0])}


def llm_reasoning(model, case):
    """The reasoning of the first recorded run that predicted the crash, if most runs did."""
    path = PATH_LLM_ANSWERS.joinpath(f"crash_detection_{model}_runinfo", f"{case}_reproduced.json")
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        answers = json.load(f)
    if not majority_detection(answers):
        return None
    return next((a.get("reasoning") or "" for a in answers if isinstance(a, dict) and a.get("detection") is True), None)


# --- replaying one notebook (runs in its own process) -------------------------


def replay_notebook(path_nb, path_fixed):
    from crane_llm.nb_extension.checker import run_checks
    from crane_llm.nb_extension.provenance import ProvenanceLog, locate_origins, static_writes
    from crane_llm.runinfo_parser.cell_executor import IPythonExecutor
    from crane_llm.runinfo_parser.notebook_runtime_extractor import NotebookRuntimeExtractor, temporarily_change_dir
    from crane_llm.nb_extension.verdict import names_mentioned
    from crane_llm.runinfo_parser.runtime_summary import extract_dependencies

    extractor = NotebookRuntimeExtractor(path_nb)
    executor = IPythonExecutor()
    shell = executor.ipython
    log = ProvenanceLog()
    executed, errors = [], 0
    with temporarily_change_dir(extractor.notebook_dir):
        for item in extractor.executed_cells_in_order():
            log.before_cell(item["code"], shell.user_ns)
            error = extractor.execute_code_cell(executor, item)
            errors += error is not None
            log.after_cell(item["code"], cell_id=str(item["code_cell_id"]), execution_count=item["execution_count"],
                           succeeded=error is None, namespace=shell.user_ns)
            executed.append(item["code_cell_id"])

        target = extractor.processed_nb["target"]
        used, _ = extract_dependencies(target["code"], shell=shell)
        used = sorted(used)
        read = [n for n in used if not isinstance(shell.user_ns.get(n), types.ModuleType)]
        sources = code_sources(path_nb)
        changed, lines, inserted, marked = diff_fix(path_nb, path_fixed)
        touched = names_in(lines).union(*(static_writes(sources[i]).stores for i in changed - {target["code_cell_id"]}))
        finding = run_checks(target["code"], shell.user_ns, shell=shell)
        variables = {
            "fix": sorted(set(read) & touched),
            "read": read,
            "checker": list(finding.variables) if finding is not None else None,
        }
        case = extractor.id_name
        for model in LLM_MODELS:
            reasoning = llm_reasoning(model, case)
            # As the extension's assistant.read_response does when the model names no variables.
            variables[f"llm_{model}"] = names_mentioned(reasoning, used) if reasoning is not None else None
        notebook_cells = [{"id": str(i), "source": s} for i, s in enumerate(sources)]
        reported = {}
        for setup, names in variables.items():
            if names is None:
                continue
            origins = locate_origins(names, log, shell.user_ns, notebook_cells=notebook_cells,
                                     target_cell_id=str(target["code_cell_id"]))
            steps = {int(step.cell_id): step.role for origin in origins for step in origin.steps}
            reported[setup] = [{"cell": cell, "role": role} for cell, role in sorted(steps.items())]

    # Changed cells that never ran matter only if they would define a name the crashing cell reads
    # (the summary also requires them to come before it, or the fix to say to execute them).
    others = changed - {target["code_cell_id"]}
    return {
        "target": target["code_cell_id"],
        "code_cells": len(sources),
        "executed": executed,
        "prior_errors": errors,
        "target_changed": target["code_cell_id"] in changed,
        "fixed_ran": sorted(others & set(executed)),
        "fixed_not_run": sorted(i for i in others - set(executed) if set(static_writes(sources[i]).stores) & set(read)),
        "fixed_other": sorted(i for i in others - set(executed) if not set(static_writes(sources[i]).stores) & set(read)),
        "marked_execute": sorted(marked),
        "inserted_cells": inserted,
        "flagged": finding is not None,
        "variables": variables,
        "reported": reported,
    }


def worker(path_nb, path_fixed, path_out):
    record = replay_notebook(Path(path_nb), Path(path_fixed))
    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(record, f, default=str)


# --- running the benchmark ---------------------------------------------------


def run_benchmark(nb_root, libs, cases, out, timeout, rerun=False):
    from crane_llm.llms.prompt_extractor import case_notebook

    path_cases, path_status = out.joinpath("cases.jsonl"), out.joinpath("notebooks.jsonl")
    out.joinpath("logs").mkdir(parents=True, exist_ok=True)
    done = set() if rerun else {r["case"] for r in read_jsonl(path_status) if r["status"] == "ok"}
    for lib, case, version, path_nb, _ in list_notebooks(nb_root, libs, cases):
        path_fixed = case_notebook(nb_root.joinpath(lib, case), "fixed")
        if version != "reproduced" or path_fixed is None or case in done:
            continue
        print(f"Replaying {case}", flush=True)
        path_record = out.joinpath("logs", f"{case}.json")
        start = time.perf_counter()
        try:
            with open(out.joinpath("logs", f"{case}.log"), "w", encoding="utf-8") as log:
                result = subprocess.run(
                    [sys.executable, os.path.abspath(__file__), "--worker",
                     str(path_nb.resolve()), str(path_fixed.resolve()), str(path_record.resolve())],
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
            status = "ok" if result.returncode == 0 and path_record.exists() else f"exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
        if status == "ok":
            with open(path_record, encoding="utf-8") as f:
                record = json.load(f)
            append_jsonl(path_cases, [{"lib": lib, "case": case, **record}])
            path_record.unlink()
        else:
            print(f"  {case}: {status}, see logs/{case}.log", flush=True)
        append_jsonl(path_status, [{"case": case, "status": status, "seconds": time.perf_counter() - start}])


# --- summarizing ---------------------------------------------------------------


def score(rows, setup, lines):
    """How often the cells reported for ``setup`` include a fixed cell, and how many are reported."""
    asked = rows[rows["reported"].map(lambda r: setup in r)]
    if asked.empty:
        return
    found_by = []
    for reported, fixed in zip(asked["reported"].map(lambda r: r[setup]), asked["fixed"]):
        roles = [step["role"] for step in reported if step["cell"] in fixed]
        found_by.append(None if not roles else "definer search" if all(r == "defines" for r in roles) else "history")
    found_by = pd.Series(found_by, index=asked.index, dtype=object)
    by_group = []
    for column, name in (("fixed_ran", "a fixed cell ran before the crash"), ("fixed_not_run", "a fixed cell never ran"),
                         ("fixed", "NBspecific cases"), ("fixed", "other cases")):
        group = asked[asked[column].map(len) > 0]
        if name == "NBspecific cases":
            group = group[group["nb_specific"]]
        elif name == "other cases":
            group = group[~group["nb_specific"]]
        hits = sum(bool({step["cell"] for step in reported[setup]} & set(fixed))
                   for reported, fixed in zip(group["reported"], group[column]))
        by_group.append(f"{name}: {ratio(hits, len(group))}")
    sizes = asked["reported"].map(lambda r: len(r[setup]))
    earlier = asked["executed"].map(len)
    no_variables = int(asked["variables"].map(lambda v: not v[setup]).sum())
    lines += [
        f"{setup}: fixed cell reported in {ratio(int(found_by.notna().sum()), len(asked))} "
        f"(through the history: {int((found_by == 'history').sum())}, "
        f"through the search for defining cells: {int((found_by == 'definer search').sum())})",
        f"  where {'; where '.join(by_group[:2])}",
        f"  in {'; in '.join(by_group[2:])}",
        f"  cells reported: median {sizes.median():.0f}, mean {sizes.mean():.1f}, max {sizes.max()}; "
        f"cells run before the crash: median {earlier.median():.0f}, mean {earlier.mean():.1f}"
        + (f"; no variable to trace: {no_variables}" if no_variables else ""),
    ]
    return found_by


def marked_execute(nb_root, lib, case):
    """The cells the fix marks as ones to execute, for records written before
    the replay recorded them. Only diffs the notebooks; nothing is run."""
    folder = nb_root.joinpath(lib, case)
    paths = folder.joinpath(f"{case}_reproduced.ipynb"), folder.joinpath(f"{case}_fixed.ipynb")
    if not all(p.exists() for p in paths):
        return []
    return sorted(diff_fix(*paths)[3])


def summarize(out, nb_root):
    rows = pd.DataFrame(read_jsonl(out.joinpath("cases.jsonl")))
    if rows.empty:
        print(f"No results in {out}")
        return
    rows = rows.drop_duplicates("case", keep="last").reset_index(drop=True)
    status = pd.DataFrame(read_jsonl(out.joinpath("notebooks.jsonl"))).drop_duplicates("case", keep="last")
    failed = status[status["status"] != "ok"]

    rows["nb_specific"] = rows["case"].str.startswith("NBspecific")
    if "marked_execute" not in rows:
        rows["marked_execute"] = None
    rows["marked_execute"] = [m if isinstance(m, list) else marked_execute(nb_root, lib, case)
                              for m, lib, case in zip(rows["marked_execute"], rows["lib"], rows["case"])]
    # A never-run cell counts only if it comes before the crashing cell or the fix says to execute it.
    keep = [[i for i in cells if counts_as_fixed_not_run(i, target, marked)]
            for cells, target, marked in zip(rows["fixed_not_run"], rows["target"], rows["marked_execute"])]
    rows["fixed_not_run_dropped"] = [sorted(set(cells) - set(k)) for cells, k in zip(rows["fixed_not_run"], keep)]
    rows["fixed_not_run"] = keep

    rows["fixed"] =[ran + not_run for ran, not_run in zip(rows["fixed_ran"], rows["fixed_not_run"])]
    # The extension asks the checker first, and the LLM only when the checker finds nothing.
    for column in ("reported", "variables"):
        def combine(values):
            values = dict(values)
            for model in LLM_MODELS:
                chosen = "checker" if values.get("checker") is not None else f"llm_{model}"
                if values.get(chosen) is not None:
                    values[f"checker+llm_{model}"] = values[chosen]
            return values
        rows[column] = rows[column].map(combine)
    scored = rows[rows["fixed"].map(len) > 0].copy()
    unscored = rows[rows["fixed"].map(len) == 0]
    lines = [
        f"Cases replayed: {len(rows)}, failed: {len(failed)}",
        f"Cases whose earlier cells all ran in this replay: {ratio(int((rows['prior_errors'] == 0).sum()), len(rows))}",
        "",
        "== What the fix changed (from the diff of the reproduced and fixed notebooks) ==",
        f"Crashing cell unchanged: {ratio(int((~rows['target_changed']).sum()), len(rows))}",
        f"Scored, the fix changed an earlier cell: {len(scored)}",
        f"  a cell that ran before the crash: {int((scored['fixed_ran'].map(len) > 0).sum())}",
        f"  a cell that never ran, which assigns a name the crashing cell reads and comes before it "
        f"or the fix says to execute: {int((scored['fixed_not_run'].map(len) > 0).sum())}",
        f"  NBspecific cases: {int(scored['nb_specific'].sum())}, other cases: {int((~scored['nb_specific']).sum())}",
        f"  never-run cells left out because they come after the crashing cell and the fix does not say to "
        f"execute them: {', '.join(rows.loc[rows['fixed_not_run_dropped'].map(len) > 0, 'case']) or 'none'}",
        f"Not scored, the fix changed only the crashing cell or cells unrelated to it: {len(unscored)}",
        f"Fix inserts new cells: {int(rows['inserted_cells'].sum())} (a new cell cannot be reported)",
        "",
        "== Is a fixed cell among the cells the tracer reports? ==",
    ]
    for setup in SETUPS + COMBINED:
        found_by = score(scored, setup, lines)
        if found_by is not None:
            scored[f"found_{setup}"] = found_by
    lines.append("")
    if len(failed):
        lines += ["Failed notebooks (left out above):", failed.to_string(index=False), ""]

    rows.to_csv(out.joinpath("cases.csv"), index=False)
    scored.to_csv(out.joinpath("scored_cases.csv"), index=False)
    text = "\n".join(lines)
    out.joinpath("summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser(description="Evaluate whether the variable tracer reports the cells a fix changed.")
    parser.add_argument("--nb-root", default="target_nbs", help="benchmark root, laid out as <lib>/<case>/<case>_reproduced.ipynb and <case>_fixed.ipynb")
    parser.add_argument("--libs", nargs="*", help="library folders to run, by default all of them")
    parser.add_argument("--cases", nargs="*", help="case names to run, e.g. pandas_1, by default all of them")
    parser.add_argument("--out", help="output folder, by default results/tracer/<name of nb-root>")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds allowed to replay one notebook")
    parser.add_argument("--rerun", action="store_true", help="replay notebooks again even if they finished before")
    parser.add_argument("--summarize-only", action="store_true", help="only summarize the results already recorded")
    parser.add_argument("--worker", nargs=3, metavar=("NOTEBOOK", "FIXED", "OUTPUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(*args.worker)
        return
    nb_root = Path(args.nb_root)
    out = Path(args.out) if args.out else Path("results/tracer").joinpath(nb_root.name)
    out.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        run_benchmark(nb_root, args.libs, set(args.cases or []), out, args.timeout, args.rerun)
    summarize(out, nb_root)


if __name__ == "__main__":
    main()
