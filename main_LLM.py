"""Run the CRANE-LLM paper's LLM pipeline on a dataset: build the prompts, then query an LLM.

Prompts: each reproduced and fixed notebook is replayed up to its target cell, in a
process of its own so that nothing carries over from the previous notebook, and its
executed cells, runtime information and target cell are written to
<llms>/llms_inputs/executed_code_runinfo_<ablation>/<lib>/<instance>.txt. Prompts that
already exist are kept.

Queries: each prompt is sent --runs times, and the answers go to
<llms>/llms_outputs/crash_detection_<model>_runinfo_<ablation>/<instance>.json. A file
that already holds enough answers is skipped, so an interrupted run resumes.

<llms> is llms/ for JunoBench and llms/<dataset>/ for any other dataset.

    python main_LLM.py --dataset target_nbs_notebookerrorsdataset      # GPT-5, full runtime information
    python main_LLM.py --model gemini-2.5-flash --ablations r_v s_r s_v
    python main_LLM.py --task code                                      # executed cells only
    python main_LLM.py --libs sklearn --cases sklearn_3 --prompts-only
"""

import argparse
import json
import os
import subprocess
import sys

from crane_llm.llms import prompt_extractor
from crane_llm.llms.config_llms import config

TASKS = {
    "runinfo": "crash detection with executed code cells and runinfo",
    "code": "crash detection with executed code cells",
}
SERVERS = {
    "gpt-5": "openai_gpt",
    "gpt-5-mini": "openai_gpt",
    "gemini-2.5-flash": "google_gemini",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "local_huggingface",
}


def setup(args):
    config.use_dataset(args.dataset)
    config.current_task = TASKS[args.task]
    config.current_llm_model = args.model


def build_prompts(lib, case, ablations):
    """Write the prompts of one case (runs in its own process)."""
    if "full" in ablations or config.current_task == TASKS["code"]:
        prompt_extractor.generate_prompt(lib, case_names=[case])
    if config.current_task == TASKS["runinfo"] and set(ablations) - {"full"}:
        prompt_extractor.generate_prompt(lib, case_names=[case], generate_all_combinations=True)


def build_all_prompts(args, libs):
    for lib in libs:
        for case in sorted(prompt_extractor.list_case_names(lib)):
            if args.cases and case not in args.cases:
                continue
            command = [sys.executable, os.path.abspath(__file__), "--prompt-worker", lib, case,
                       "--dataset", args.dataset, "--task", args.task, "--ablations", *args.ablations]
            try:
                result = subprocess.run(command, timeout=args.timeout, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                if result.returncode != 0:
                    print(f"  {case}: prompts failed, exit code {result.returncode}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"  {case}: prompts timed out", flush=True)


def answers_so_far(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return len(json.load(f))


def query_all(args, libs):
    from crane_llm.llms import llm_executor

    server = SERVERS[args.model]
    if server == "local_huggingface":
        from crane_llm.llms.huggingface_model_loader import get_qwen_model
        tokenizer, model = get_qwen_model(args.model)
    for ablation in args.ablations if args.task == "runinfo" else ["full"]:
        config.current_ablation_setting = ablation
        print(f"{config.current_task}, {ablation}, {args.model}: answers to {config.path_res}", flush=True)
        for run in range(1, args.runs + 1):
            for lib in libs:
                path_lib = config.path_input.joinpath(lib)
                if not path_lib.exists():
                    continue
                for filename in sorted(os.listdir(path_lib)):
                    instance = os.path.splitext(filename)[0]
                    if args.cases and not any(instance.startswith(f"{case}_") for case in args.cases):
                        continue
                    if answers_so_far(config.path_res.joinpath(f"{instance}.json")) >= run:
                        continue
                    executor = llm_executor.LLMExecutor(model=args.model, libname=lib, filename=filename)
                    if server == "openai_gpt":
                        executor.llm_multiple_rounds_openai()
                    elif server == "google_gemini":
                        executor.llm_multiple_rounds_gemini()
                    else:
                        executor.llm_multiple_rounds_huggingface(tokenizer, model)
            print(f"Run {run} of {args.runs} done", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build the prompts of a dataset and query an LLM on them.")
    parser.add_argument("--dataset", default="target_nbs", help="benchmark root, laid out as target_nbs/ (JunoBench, the default)")
    parser.add_argument("--model", default="gpt-5", choices=list(SERVERS))
    parser.add_argument("--task", default="runinfo", choices=list(TASKS), help="executed cells with runtime information, or executed cells only")
    parser.add_argument("--ablations", nargs="+", default=["full"], choices=["full", "s_r", "s_v", "r_v"],
                        help="which runtime information to give, for --task runinfo: all of it, or leave out one category")
    parser.add_argument("--runs", type=int, default=5, help="answers to collect per prompt")
    parser.add_argument("--libs", nargs="*", help="library folders to run, by default all of them")
    parser.add_argument("--cases", nargs="*", help="case names to run, e.g. pandas_1, by default all of them")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds allowed to build the prompts of one case")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prompts-only", action="store_true", help="build the prompts and stop")
    stage.add_argument("--skip-prompts", action="store_true", help="only query, on the prompts already built")
    parser.add_argument("--prompt-worker", nargs=2, metavar=("LIB", "CASE"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    setup(args)
    if args.prompt_worker:
        build_prompts(*args.prompt_worker, args.ablations)
        return
    libs = args.libs or prompt_extractor.list_lib_names()
    if not args.skip_prompts:
        build_all_prompts(args, libs)
    if not args.prompts_only:
        query_all(args, libs)


if __name__ == "__main__":
    main()
