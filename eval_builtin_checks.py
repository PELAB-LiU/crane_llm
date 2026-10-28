"""Evaluate the built-in checker and the check before each cell runs.

Each notebook of the benchmark is replayed in execution order, as in the
paper's experiments. Before every cell, the built-in checker looks at it with
the live kernel state, exactly as the guard does when it is switched on; the
cell then runs, and whether it raised is recorded. At the end, the target cell
is checked but not run: the benchmark already says whether it crashes.

- Target cells: how many crashes the checker finds in the reproduced
  notebooks, whether it reports the right exception and line, and whether it
  ever flags the target of a fixed notebook.
- All earlier cells: every one of them ran without error in the original
  notebook, so a flag on a cell that then runs fine is a cell the guard would
  have stopped for nothing. Also the time each check adds to a cell.
- Checker plus LLM: recorded LLM answers, with the checker answering first
  wherever it finds a crash, as the extension does. For JunoBench, the answers
  the paper recorded; for another dataset, those main_LLM.py recorded on it,
  if any. No LLM is called.

Each notebook runs in a process of its own, so that no variable, import or
working directory carries over from the previous one. Finished notebooks are
skipped when the script is run again, so an interrupted run can be resumed;
--rerun replays them anyway, and the new results replace the old.

    python eval_builtin_checks.py                           # JunoBench, all libraries
    python eval_builtin_checks.py --libs pandas --cases pandas_1
    python eval_builtin_checks.py --nb-root target_nbs_notebookerrorsdataset  # another dataset
    python eval_builtin_checks.py --libs sklearn --cases sklearn_3 --rerun
    python eval_builtin_checks.py --summarize-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

VERSIONS = ["reproduced", "fixed"]
LLM_MODELS = ["gpt5", "gemini", "qwen"]
PATH_LLM_LABELS = Path("results/results_parsed_detection_only.xlsx")
PATH_LLM_QUERY_TIMES =Path("results/runtime_recording/runtime_counts_cranellm_gpt5.json")
PATH_GPT5_INPUTS = Path("llms/llms_inputs/executed_code_runinfo_full")
PATH_GPT5_OUTPUTS = Path("llms/llms_outputs/results_raw/crash_detection_gpt5_runinfo")
# GPT-5 list prices in USD per million tokens, input and output.
GPT5_PRICES = (1.25, 10.0)


# --- replaying one notebook (runs in its own process) -------------------------


def check_cell(source, shell, before_run):
    """Run the built-in checker on a cell, as the guard or the toolbar button would."""
    from crane_llm.nb_extension.checker import run_checks
    from crane_llm.nb_extension.guard import checks_cell

    if before_run and not checks_cell(source):
        return {"checked": False, "flagged": False}
    start = time.perf_counter()
    finding = run_checks(source, shell.user_ns, shell=shell)
    record = {"checked": True, "check_seconds": time.perf_counter() - start, "flagged": finding is not None}
    if finding is not None:
        record.update({f"finding_{key}": value for key, value in asdict(finding).items()})
    return record


def replay_notebook(path_nb):
    from crane_llm.runinfo_parser.cell_executor import IPythonExecutor
    from crane_llm.runinfo_parser.notebook_runtime_extractor import NotebookRuntimeExtractor, temporarily_change_dir

    extractor = NotebookRuntimeExtractor(path_nb)
    executor = IPythonExecutor()
    shell = executor.ipython
    records = []
    with temporarily_change_dir(extractor.notebook_dir):
        for item in extractor.executed_cells_in_order():
            record = {"role": "prior", "code_cell_id": item["code_cell_id"], **check_cell(item["code"], shell, before_run=True)}
            start = time.perf_counter()
            error = extractor.execute_code_cell(executor, item)
            record["run_seconds"] = time.perf_counter() - start
            record["run_error"] = type(error).__name__ if error is not None else None
            records.append(record)
        target = extractor.processed_nb["target"]
        records.append({"role": "target", "code_cell_id": target["code_cell_id"], **check_cell(target["code"], shell, before_run=False)})
    return records


def worker(path_nb, path_out):
    records = replay_notebook(Path(path_nb))
    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(records, f, default=str)


# --- running the benchmark ---------------------------------------------------


def list_notebooks(nb_root, libs, cases):
    """Each notebook to replay, with the reproduced notebook of its case, which holds the crash."""
    from crane_llm.llms.prompt_extractor import case_notebook, list_case_names, list_lib_names

    for lib in libs or list_lib_names(nb_root):
        for case in sorted(list_case_names(lib, nb_root)):
            if cases and case not in cases:
                continue
            path_case = nb_root.joinpath(lib, case)
            path_reproduced = case_notebook(path_case, "reproduced")
            for version in VERSIONS:
                path_nb = case_notebook(path_case, version)
                if path_nb is not None:
                    yield lib, case, version, path_nb, path_reproduced


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def run_benchmark(nb_root, libs, cases, out, timeout, rerun=False):
    from crane_llm.runinfo_parser.preprocess_notebook import describe_crash

    path_cells, path_status = out.joinpath("cells.jsonl"), out.joinpath("notebooks.jsonl")
    out.joinpath("logs").mkdir(parents=True, exist_ok=True)
    done = set() if rerun else {r["instance"] for r in read_jsonl(path_status) if r["status"] == "ok"}
    for lib, case, version, path_nb, path_reproduced in list_notebooks(nb_root, libs, cases):
        instance = f"{case}_{version}"
        if instance in done:
            continue
        print(f"Replaying {instance}", flush=True)
        path_records = out.joinpath("logs", f"{instance}.json")
        start = time.perf_counter()
        try:
            with open(out.joinpath("logs", f"{instance}.log"), "w", encoding="utf-8") as log:
                result = subprocess.run(
                    [sys.executable, os.path.abspath(__file__), "--worker", str(path_nb.resolve()), str(path_records.resolve())],
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
            status = "ok" if result.returncode == 0 and path_records.exists() else f"exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
        seconds = time.perf_counter() - start
        if status == "ok":
            crash = describe_crash(str(path_reproduced)) or {}
            with open(path_records, encoding="utf-8") as f:
                records = json.load(f)
            for record in records:
                record.update({"lib": lib, "case": case, "version": version, "instance": instance})
                if record["role"] == "target":
                    record["crashes"] = version == "reproduced"
                    if record["crashes"]:
                        record["crash_exception"] = crash.get("exception")
                        record["crash_line"] = crash.get("line")
            append_jsonl(path_cells, records)
            path_records.unlink()
        else:
            print(f"  {instance}: {status}, see logs/{instance}.log", flush=True)
        append_jsonl(path_status, [{"instance": instance, "status": status, "seconds": seconds}])


# --- summarizing ---------------------------------------------------------------


def latest_records(out):
    """One record per cell, from the latest run of each notebook."""
    cells = pd.DataFrame(read_jsonl(out.joinpath("cells.jsonl")))
    if cells.empty:
        return cells
    return cells.drop_duplicates(subset=["instance", "role", "code_cell_id"], keep="last")


def seconds_stats(series):
    series = series.dropna()
    return (f"median {series.median() * 1000:.1f} ms, p95 {series.quantile(0.95) * 1000:.1f} ms, "
            f"max {series.max() * 1000:.1f} ms (n={len(series)})")


def ratio(n, d):
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else f"{n}/0"


def crash_metrics(truth, predicted):
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    tn = int((~truth & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / len(truth), "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def summarize_targets(targets, lines):
    crashing, fixed = targets[targets["crashes"]], targets[~targets["crashes"]]
    found = crashing[crashing["flagged"]]
    tp, fp = len(found), int(fixed["flagged"].sum())
    lines += [
        "== Target cells ==",
        # An earlier cell that fails here leaves a state the original notebook never had.
        f"Targets whose earlier cells all ran in this replay: {ratio(int((targets['prior_errors'] == 0).sum()), len(targets))}",
        f"Crashes found (reproduced targets flagged): {ratio(tp, len(crashing))}",
    ]
    # Without fixed notebooks there are no targets that should not be flagged.
    if len(fixed):
        lines += [f"False positives (fixed targets flagged): {ratio(fp, len(fixed))}", f"Precision: {ratio(tp, tp + fp)}"]
    lines += [
        f"Right exception, of crashes found: {ratio(int((found['finding_exception'] == found['crash_exception']).sum()), tp)}",
        f"Right line, of crashes found with a known line: "
        f"{ratio(int((found['finding_line'] == found['crash_line']).sum()), int(found['crash_line'].notna().sum()))}",
        f"Check time: {seconds_stats(targets['check_seconds'])}",
        "",
        "Crashes found by library:",
        crashing.groupby("lib")["flagged"].agg(found="sum", cases="count").to_string(),
        "",
    ]
    if tp:
        lines += ["Crashes found by rule:", found["finding_rule"].value_counts().to_string(), ""]
        mismatched = found[found["finding_exception"] != found["crash_exception"]]
        if len(mismatched):
            lines += ["Crashes found with another exception than the notebook's:",
                      mismatched[["instance", "finding_rule", "finding_exception", "crash_exception"]].to_string(index=False), ""]
    if fp:
        lines += ["False positives:",
                  fixed[fixed["flagged"]][["instance", "finding_rule", "finding_exception", "finding_message"]].to_string(index=False), ""]


def summarize_prior_cells(prior, lines):
    checked = prior[prior["checked"]]
    flagged = checked[checked["flagged"]]
    false_flags = flagged[flagged["run_error"].isna()]
    raised = checked[checked["run_error"].notna()]
    ran = checked.dropna(subset=["check_seconds", "run_seconds"])
    lines += [
        "== Cells before the target (the check before each cell runs) ==",
        f"Cells checked: {len(checked)} of {len(prior)} (cell magics are not checked), "
        f"in the {prior['instance'].nunique()} notebooks that run any cell before the target",
        f"Would have been stopped although they ran fine: {ratio(len(false_flags), int(checked['run_error'].isna().sum()))}",
        f"Raised in this replay (the environment differs from the original): {len(raised)}, "
        f"of which flagged {ratio(int(raised['flagged'].sum()), len(raised))}",
        f"Check time per cell: {seconds_stats(checked['check_seconds'])}",
        f"Cell run time: {seconds_stats(checked['run_seconds'])}",
        f"Check time relative to run time, per cell: median {(ran['check_seconds'] / ran['run_seconds']).median():.1%}",
        f"Total check time relative to total run time: {ran['check_seconds'].sum() / ran['run_seconds'].sum():.2%}",
        "",
    ]
    if len(false_flags):
        lines += ["Cells that would have been stopped although they ran fine:",
                  false_flags[["instance", "code_cell_id", "finding_rule", "finding_exception", "finding_message"]].to_string(index=False), ""]


def summarize_all_cells(targets, prior, lines):
    """Precision over every cell the checker looked at. A flag is right on a crashing
    target, and wrong on a fixed target or on an earlier cell that ran fine; earlier
    cells that raised in this replay count as neither."""
    tp = int((targets["crashes"] & targets["flagged"]).sum())
    fp = int((~targets["crashes"] & targets["flagged"]).sum())
    fp += int((prior["checked"] & prior["flagged"] & prior["run_error"].isna()).sum())
    lines += ["== All checked cells ==", f"Precision: {ratio(tp, tp + fp)}", ""]


def is_junobench(nb_root):
    return nb_root.resolve() == Path("target_nbs").resolve()


def paper_llm_predictions():
    """Whether each model predicted a crash in each target, from the labels the paper's results are computed on."""
    labels = pd.read_excel(PATH_LLM_LABELS, sheet_name="Final_evaluation")
    # The sheet ends with rows of totals below the instances.
    labels = labels[labels["instance"].astype(str).str.match(r"^\w+_\d+_(fixed|reproduced)$")].set_index("instance")
    predictions = {}
    for model in LLM_MODELS:
        column = f"crash_detection_{model}_runinfo"
        if column not in labels:
            continue
        correct = labels[column] == "correct"
        crashes = labels.index.str.endswith("_reproduced")
        predictions[model] = pd.Series(correct == crashes, index=labels.index)
    return predictions


def llm_answers_folder(nb_root, model):
    """Where main_LLM.py records a model's answers on a dataset, with all the runtime information."""
    from crane_llm.llms.config_llms import config

    config.use_dataset(nb_root)
    return config.path_llms.joinpath("llms_outputs", f"crash_detection_{model}_runinfo_full")


def majority_detection(answers):
    """Whether most runs predicted a crash; on a tie, what the first run said.
    An answer that is not valid JSON with a boolean "detection" is left out."""
    votes = [a["detection"] for a in answers if isinstance(a, dict) and isinstance(a.get("detection"), bool)]
    if not votes:
        return False
    yes = sum(votes)
    return votes[0] if 2 * yes == len(votes) else 2 * yes > len(votes)


def recorded_llm_predictions(nb_root):
    """Whether each model predicted a crash in each target, by majority over the runs main_LLM.py recorded."""
    predictions = {}
    for model in LLM_MODELS:
        folder = llm_answers_folder(nb_root, model)
        answers = {}
        for path in sorted(folder.glob("*.json")):
            with open(path, encoding="utf-8") as f:
                answers[path.stem] = majority_detection(json.load(f))
        if answers:
            predictions[model] = pd.Series(answers, dtype=bool)
    return predictions


def summarize_with_llm(targets, lines, nb_root):
    junobench = is_junobench(nb_root)
    predictions = paper_llm_predictions() if junobench else recorded_llm_predictions(nb_root)
    if not predictions:
        return
    source = "the paper's recorded answers" if junobench else "the answers main_LLM.py recorded, majority over the runs"
    lines += [f"== Checker first, then the LLM ({source}, CRANE-LLM with runtime information) =="]
    for model, predicted in predictions.items():
        rows = targets[targets["instance"].isin(predicted.index)]
        if rows.empty:
            continue
        truth = rows["crashes"].astype(bool).to_numpy()
        llm = predicted.loc[rows["instance"]].to_numpy()
        both = llm | rows["flagged"].to_numpy()
        for name, values in (("LLM only", llm), ("checker + LLM", both)):
            m = crash_metrics(truth, values)
            if truth.all():  # without targets that should not be flagged, only recall means anything
                lines.append(f"{model:7} {name:14} recall {m['recall']:.3f}  (TP {m['tp']} FN {m['fn']}, n={len(truth)})")
            else:
                lines.append(f"{model:7} {name:14} accuracy {m['accuracy']:.3f}  precision {m['precision']:.3f}  "
                             f"recall {m['recall']:.3f}  F1 {m['f1']:.3f}  (TP {m['tp']} FP {m['fp']} FN {m['fn']} TN {m['tn']}, n={len(truth)})")
    flagged = targets[targets["flagged"]]
    lines.append(f"LLM calls avoided: {ratio(len(flagged), len(targets))} target cells answered by the checker alone")
    if junobench and PATH_LLM_QUERY_TIMES.exists():
        with open(PATH_LLM_QUERY_TIMES, encoding="utf-8") as f:
            query_times = {name[:-len(".txt")]: t for lib in json.load(f).values() for name, t in lib.items()}
        saved = [query_times[i] for i in flagged["instance"] if i in query_times]
        if saved:
            lines.append(f"GPT-5 query time of those cells: median {pd.Series(saved).median():.2f} s, "
                         f"against a checker median of {flagged['check_seconds'].median() * 1000:.1f} ms")
    if junobench:
        summarize_gpt5_cost(targets, lines, PATH_GPT5_INPUTS, PATH_GPT5_OUTPUTS)
    else:
        from crane_llm.llms.config_llms import config

        summarize_gpt5_cost(targets, lines, config.path_input_executed_code_runinfo, llm_answers_folder(nb_root, "gpt5"))
    lines.append("")


def gpt5_tokens(targets, path_inputs, path_outputs):
    """The GPT-5 tokens of one query per target: the prompt sent, and the mean
    of the answers recorded. GPT-5's reasoning tokens were not
    recorded, so the output is a lower bound."""
    import tiktoken
    from crane_llm.llms.config_llms import config
    from utils.token_analysis import count_tokens

    tokenizer = tiktoken.get_encoding("o200k_base")
    system = count_tokens(config.prompt_instruct_crash_detection_1_enforcejson, tokenizer)
    rows = []
    for lib, instance in targets[["lib", "instance"]].itertuples(index=False):
        path_in, path_out = path_inputs.joinpath(lib, f"{instance}.txt"), path_outputs.joinpath(f"{instance}.json")
        if not path_in.exists() or not path_out.exists():
            continue
        with open(path_out, encoding="utf-8") as f:
            answers = json.load(f)
        rows.append({
            "instance": instance,
            "input": system + count_tokens(path_in.read_text(encoding="utf-8"), tokenizer),
            "output": sum(count_tokens(json.dumps(a), tokenizer) for a in answers) / max(len(answers), 1),
        })
    return pd.DataFrame(rows, columns=["instance", "input", "output"]).set_index("instance")


def summarize_gpt5_cost(targets, lines, path_inputs, path_outputs):
    try:
        tokens = gpt5_tokens(targets, path_inputs, path_outputs)
    except ImportError as exc:
        lines.append(f"GPT-5 cost left out: {exc}")
        return
    if tokens.empty:
        return
    avoided = tokens[tokens.index.isin(targets.loc[targets["flagged"], "instance"])]
    price_in, price_out = GPT5_PRICES

    def cost(rows):
        return (rows["input"].sum() * price_in + rows["output"].sum() * price_out) / 1e6

    lines += [
        f"GPT-5 tokens per query: median input {tokens['input'].median():.0f}, output {tokens['output'].median():.0f} "
        f"(the recorded answer only; reasoning tokens were not recorded)",
        f"GPT-5 input tokens avoided: {ratio(int(avoided['input'].sum()), int(tokens['input'].sum()))}",
        f"GPT-5 cost avoided per pass over the {len(tokens)} targets, at ${price_in}/${price_out} per million "
        f"input/output tokens: ${cost(avoided):.4f} of ${cost(tokens):.4f} ({cost(avoided) / cost(tokens):.1%})",
    ]


def summarize(out, nb_root, with_llm):
    cells = latest_records(out)
    if cells.empty:
        print(f"No results in {out}")
        return
    status = pd.DataFrame(read_jsonl(out.joinpath("notebooks.jsonl"))).drop_duplicates("instance", keep="last")
    failed = status[status["status"] != "ok"]
    lines = [f"Notebooks replayed: {int((status['status'] == 'ok').sum())}, failed: {len(failed)}", ""]
    if len(failed):
        lines += ["Failed notebooks (left out below):", failed.to_string(index=False), ""]

    # Columns that only exist once the checker has flagged something.
    for column in ("finding_rule", "finding_exception", "finding_message", "finding_line", "crash_exception", "crash_line"):
        if column not in cells:
            cells[column] = None
    prior = cells[cells["role"] == "prior"].copy()
    prior[["checked", "flagged"]] = prior[["checked", "flagged"]].astype(bool)
    targets = cells[cells["role"] == "target"].copy()
    targets["crashes"] = targets["crashes"].astype(bool)
    targets["flagged"] = targets["flagged"].astype(bool)
    prior_errors = prior.groupby("instance")["run_error"].count()
    targets["prior_errors"] = targets["instance"].map(prior_errors).fillna(0).astype(int)
    summarize_targets(targets, lines)
    summarize_prior_cells(prior, lines)
    summarize_all_cells(targets, prior, lines)
    if with_llm:
        summarize_with_llm(targets, lines, nb_root)

    targets.to_csv(out.joinpath("target_cells.csv"), index=False)
    prior.to_csv(out.joinpath("prior_cells.csv"), index=False)
    text = "\n".join(lines)
    out.joinpath("summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the built-in checker on a benchmark of crashing notebooks.")
    parser.add_argument("--nb-root", default="target_nbs", help="benchmark root, laid out as <lib>/<case>/<case>_reproduced[_1].ipynb, optionally with <case>_fixed.ipynb")
    parser.add_argument("--libs", nargs="*", help="library folders to run, by default all of them")
    parser.add_argument("--cases", nargs="*", help="case names to run, e.g. pandas_1, by default all of them")
    parser.add_argument("--out", help="output folder, by default results/builtin_checks/<name of nb-root>")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds allowed to replay one notebook")
    parser.add_argument("--rerun", action="store_true", help="replay notebooks again even if they finished before; the new results replace the old")
    parser.add_argument("--summarize-only", action="store_true", help="only summarize the results already recorded")
    parser.add_argument("--no-llm", action="store_true", help="leave out the comparison with the recorded LLM answers")
    parser.add_argument("--worker", nargs=2, metavar=("NOTEBOOK", "OUTPUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(*args.worker)
        return
    nb_root = Path(args.nb_root)
    out = Path(args.out) if args.out else Path("results/builtin_checks").joinpath(nb_root.name)
    out.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        run_benchmark(nb_root, args.libs, set(args.cases or []), out, args.timeout, args.rerun)
    summarize(out, nb_root, with_llm=not args.no_llm)


if __name__ == "__main__":
    main()
