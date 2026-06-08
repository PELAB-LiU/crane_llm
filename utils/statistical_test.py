"""Statistical significance testing for crash detection settings."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path

import pandas as pd

WORKBOOK_PATH_DIAGNOSIS = Path("results/results_parsed_detection_and_diagnosis.xlsx")
WORKBOOK_PATH_DETECTION_ONLY = Path("results/results_parsed_detection_only.xlsx")
DEFAULT_SHEET_NAME = "Final_evaluation"
DEFAULT_MAX_ROWS = 223
DEFAULT_MODELS = ("gemini", "qwen", "gpt5")
DEFAULT_SETTINGS = ("code", "runinfo")

def _resolve_sheet_name(workbook_path: Path, sheet_name: str) -> str:
    workbook = pd.ExcelFile(workbook_path, engine="openpyxl")
    if sheet_name in workbook.sheet_names:
        return sheet_name

    normalized_requested = sheet_name.replace(" ", "_").lower()
    for actual_sheet_name in workbook.sheet_names:
        normalized_actual = actual_sheet_name.replace(" ", "_").lower()
        if normalized_actual == normalized_requested:
            return actual_sheet_name

    raise KeyError(f"Worksheet '{sheet_name}' not found in {workbook_path}")

def _is_instance_row(instance: object) -> bool:
    if not isinstance(instance, str):
        return False
    instance = instance.strip()
    return instance.endswith("_fixed") or instance.endswith("_reproduced")


def collect_results(
    workbook_path: Path,
    model: str,
    setting: str,
    sheet_name: str = DEFAULT_SHEET_NAME,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> list[dict]:
    """Collect one binary result per instance for a model/setting pair."""

    resolved_sheet_name = _resolve_sheet_name(workbook_path, sheet_name)
    df = pd.read_excel(
        workbook_path,
        sheet_name=resolved_sheet_name,
        engine="openpyxl",
        nrows=max_rows,
    )
    df.columns = [str(column).strip() for column in df.columns]

    if "instance" not in df.columns:
        raise KeyError("The workbook is missing the 'instance' column.")

    column_name = f"crash_detection_{model}_{setting}"
    if column_name not in df.columns:
        raise KeyError(f"The workbook is missing the '{column_name}' column.")

    filtered = df[df["instance"].map(_is_instance_row)].copy()
    results: list[dict] = []

    for _, row in filtered.iterrows():
        instance = str(row["instance"]).strip()
        raw_value = str(row[column_name]).strip().lower()
        if raw_value not in {"correct", "wrong"}:
            continue

        results.append(
            {
                "instance": instance,
                "is_correct": raw_value == "correct",
                "model": model,
                "setting": setting,
                "raw_value": raw_value,
            }
        )

    return results


def _exact_mcnemar_p_value(n10: int, n01: int) -> float:
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    lower_tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * lower_tail)


def _instance_performance_from_results(results: list[dict]) -> dict[str, int]:
    by_instance: dict[str, list[bool]] = {}
    for result in results:
        instance = result.get("instance", "")
        by_instance.setdefault(instance, []).append(bool(result.get("is_correct", False)))
    return {instance: int(any(run_correctness)) for instance, run_correctness in by_instance.items()}


def compare_settings_performance(
    setting_to_results: dict[str, list[dict]],
    alpha: float = 0.05,
) -> dict[str, object]:
    comparisons: list[dict] = []

    for setting_a, setting_b in combinations(sorted(setting_to_results.keys()), 2):
        outcomes_a = _instance_performance_from_results(setting_to_results[setting_a])
        outcomes_b = _instance_performance_from_results(setting_to_results[setting_b])
        common_instances = sorted(set(outcomes_a.keys()) & set(outcomes_b.keys()))

        n11 = n10 = n01 = n00 = 0
        for instance in common_instances:
            a = outcomes_a[instance]
            b = outcomes_b[instance]
            if a == 1 and b == 1:
                n11 += 1
            elif a == 1 and b == 0:
                n10 += 1
            elif a == 0 and b == 1:
                n01 += 1
            else:
                n00 += 1

        n_instances = len(common_instances)
        p_value = _exact_mcnemar_p_value(n10, n01)
        delta = ((n10 - n01) / n_instances) if n_instances else 0.0

        comparisons.append(
            {
                "setting_a": setting_a,
                "setting_b": setting_b,
                "n_instances": n_instances,
                "contingency_table": {
                    "n11_both_pass": n11,
                    "n10_a_pass_b_fail": n10,
                    "n01_a_fail_b_pass": n01,
                    "n00_both_fail": n00,
                },
                "discordant_pairs": n10 + n01,
                "delta_pass_rate_a_minus_b": delta,
                "mcnemar_exact_p_value": p_value,
                "significant_at_0_05": p_value < alpha,
            }
        )

    return {
        "alpha": alpha,
        "method": "Exact McNemar (two-sided), paired by instance outcomes",
        "comparisons": comparisons,
    }


def collect_results_for_setting_model(
    workbook_path: Path,
    model: str,
    settings: list[str],
    sheet_name: str = DEFAULT_SHEET_NAME,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, list[dict]]:
    setting_to_results: dict[str, list[dict]] = {}
    for setting in settings:
        setting_to_results[setting] = collect_results(
            workbook_path=workbook_path,
            model=model,
            setting=setting,
            sheet_name=sheet_name,
            max_rows=max_rows,
        )
    return setting_to_results


def run_pairwise_significance(
    workbook_path: Path,
    output_path: Path,
    models: tuple[str, ...] = DEFAULT_MODELS,
    settings: tuple[str, ...] = DEFAULT_SETTINGS,
    sheet_name: str = DEFAULT_SHEET_NAME,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> None:
    report = {
        "workbook_path": str(workbook_path),
        "sheet_name": sheet_name,
        "max_rows": max_rows,
        "settings": list(settings),
        "alpha": 0.05,
        "method": "Exact McNemar (two-sided), paired by instance outcomes",
        "comparisons_by_model": {},
    }

    for model in models:
        setting_to_results = collect_results_for_setting_model(
            workbook_path=workbook_path,
            model=model,
            settings=list(settings),
            sheet_name=sheet_name,
            max_rows=max_rows,
        )
        report["comparisons_by_model"][model] = compare_settings_performance(
            setting_to_results=setting_to_results,
            alpha=0.05,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved pairwise performance significance: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run McNemar significance tests for crash detection settings.")
    parser.add_argument(
        "--workbook",
        type=Path,
        default=WORKBOOK_PATH_DIAGNOSIS, #WORKBOOK_PATH_DETECTION_ONLY, # WORKBOOK_PATH_DIAGNOSIS,
        help="Path to results excel file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pairwise_significance_detection_and_diagnosis.json"), # pairwise_significance_detection_only.json # pairwise_significance_detection_and_diagnosis
        help="Path to the JSON report to write.",
    )
    args = parser.parse_args()

    run_pairwise_significance(
        workbook_path=args.workbook,
        output_path=args.output
    )


if __name__ == "__main__":
    main()