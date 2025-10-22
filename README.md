# CRANE-LLM: Runtime-Augmented LLMs for Crash Detection and Reasoning in ML Notebooks

This is the official repository for our paper "Runtime-Augmented LLMs for Crash Detection and Reasoning in ML Notebooks". It is an approach that augments LLMs with runtime information extracted from the notebook kernel state to enhance their detection and explanation of ML notebook crashes.

## Environment setup

To ensure full reproducibility, we provide a docker image:
```
docker pull yarinamomo/crane_env:latest
```
Then run the docker container:
```
docker run -v [volumn_mount_windows_path]:/cranellm_env -w /cranellm_env -p 8888:8888 -it yarinamomo/crane_env:latest /bin/bash
```

For the commercial LLMs used in the experiments (Gemini and GPT-5), please ensure that the API keys are properly set up before running the scripts. Open-source LLMs (Qwen-32B and Selene-8B) can be run directly; however, note that execution may take longer depending on the computational resources available.

## Repository structure
We use [**Junobench**]((https://huggingface.co/datasets/PELAB-LiU/JunoBench)) benchmark dataset in our experiments.
- [main_LLM.py](./main_LLM.py): script to run the CRANE-LLM pipeline
- [runinfo_parser](./runinfo_parser): scripts for **runtime information extraction**
- [llms](./llms): LLM-related experiments
    - `llms_inputs/`: generated inputs (executed code cells only, executed code cells with runtime information) to the LLMs
    - `llms_outputs/`: generated outputs by the LLMs
        - `results_raw/`: crash prediction outputs from the three LLMs
        - `results_parsed../`: classification on crash prediction outputs by the LLM judge
        - `ground_truth_crash_prediction.xlsx`: ground truth labels used for LLM judge for classification, provided by JunoBench
    - `prompt_extractor.py`: script for constructing prompts (i.e., `llms_inputs/`)
    - `llm_executor.py`: script for querying LLMs to generate outputs in `llms_outputs/`
- [main_sa.py](./main_sa.py): script to run the experiment pipeline involving static analyzers (SAs)
- [sas](./sas): SA-related experiments
    - `sas_inputs/executed_code_runinfo`: type annotated inputs to the SAs
    - `sas_outputs/`: generated outputs by the SAs
        - `results_raw/`: crash prediction outputs from the two SAs
        - `results_parsed../`: classification on crash prediction outputs by the LLM judge
    - `runtime_injector.py`: script for constructing type-annotated inputs (i.e., `sas_inputs/executed_code_runinfo`)
    - `static_analysis_tools.py`: script for querying SAs to generate outputs in `sas_outputs/`
- [results_summary.xlsx](./results/results_summary.xlsx): final revised results and compiled statictics

## License

This project is licensed under the terms of the BSD 3-Clause License.
