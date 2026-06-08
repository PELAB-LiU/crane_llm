import argparse
import os, json
from llms import prompt_extractor
from llms.config_llms import config
from llms import llm_executor

parser = argparse.ArgumentParser(description="Run crash detection for a reproduced case.")
parser.add_argument(
    "--nb-name",
    default="tensorflow_11_reproduced",
    help="Reproduced case name to execute, e.g., tensorflow_11_reproduced",
)
parser.add_argument(
    "--version-name",
    default="reproduced",
    help="Version suffix used when generating prompts, e.g., reproduced",
)
args = parser.parse_args()

raw_case_name = args.nb_name
version_name = args.version_name
version_suffix = f"_{version_name}"

if raw_case_name.endswith(version_suffix):
    prompt_case_name = raw_case_name[: -len(version_suffix)]
    target_case_name = raw_case_name
else:
    prompt_case_name = raw_case_name
    target_case_name = f"{raw_case_name}{version_suffix}"

current_task = "crash detection with executed code cells and runinfo"
config.current_task = current_task
print(f"Current task: {config.current_task}")

# rerun the following cases only
lib_case_names = {
    "tensorflow": [prompt_case_name],
}
for lib_name, case_names in lib_case_names.items():
    prompt_extractor.generate_prompt(lib_name, case_names=case_names, version_names=[version_name])

# settings
llm_server = "openai_gpt" # "openai_gpt", "google_gemini", "local_huggingface"
llm_model = "gpt-5" # "gpt-5", "gemini-2.5-flash", "Qwen/Qwen2.5-Coder-32B-Instruct"
config.current_llm_model = llm_model
print(f"Current LLM model: {config.current_llm_model}")

lib_case_names = {
    "tensorflow": [target_case_name],
}

for lib_name, case_names in lib_case_names.items():
    for filename in os.listdir(config.path_input.joinpath(lib_name)):
        if filename.split(".")[0] not in case_names:
            continue
        exec_llm = llm_executor.LLMExecutor(model=llm_model, libname = lib_name, filename=filename)
        if llm_server == "openai_gpt":
            exec_llm.llm_cmd_openai()
print(f"Detection completed.")