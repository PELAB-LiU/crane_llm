# CRANE-LLM: Runtime-Augmented LLMs for Crash Detection and Diagnosis in ML Notebooks

This is the official repository for our paper "Runtime-Augmented LLMs for Crash Detection and Diagnosis in ML Notebooks". It is an approach that augments LLMs with runtime information extracted from the notebook kernel state to enhance their detection and explanation of ML notebook crashes.

## Environment setup

To ensure full reproducibility, we provide a docker image:
```bash
docker pull yarinamomo/crane_env:latest
```
Then run the docker container:
```bash
docker run -v [volumn_mount_windows_path]:/cranellm_env -w /cranellm_env -p 8888:8888 -it yarinamomo/crane_env:latest /bin/bash
```
Then you can attach this environment to **VS Code** "*Dev Containers: Attach to Running Container...*"

For the commercial LLMs used in the experiments (Gemini and GPT-5), please ensure that the API keys are properly set up before running the scripts. Open-source LLMs (Qwen-32B and Selene-8B) can be run directly; however, note that execution may take longer depending on the computational resources available.

## Repository structure and reproducibility details
We use [**Junobench**]((https://huggingface.co/datasets/PELAB-LiU/JunoBench)) benchmark dataset in our experiments.
- [`main_LLM.py`](./main_LLM.py): script to run the CRANE-LLM pipeline. You can uncomment the step you wish to execute.
- [`runinfo_parser`](./runinfo_parser): scripts for **runtime information extraction**
- [`llms`](./llms): LLM-related experiments
    - [`llms_inputs/`](./llms/llms_inputs): generated inputs (executed code cells only, executed code cells with runtime information) to the LLMs
    - [`llms_outputs/`](./llms/llms_outputs): generated outputs by the LLMs
        - [`results_raw/`](./llms/llms_outputs/results_raw/): crash prediction outputs from the three LLMs
        - [`ground_truth_crash_prediction.xlsx`](./llms/llms_outputs/ground_truth_crash_prediction.xlsx): ground truth labels used for evaluating crash detection and disgnosis predictions as well as downstream analysis, provided by JunoBench
    - [`prompt_extractor.py`](./llms/prompt_extractor.py): script for constructing prompts (i.e., `llms_inputs/`)
    - [`llm_executor.py`](./llms/llm_executor.py): script for querying LLMs to generate outputs in `llms_outputs/`
- [`results`](./results): results and compiled statistics
    - [`results_parsed_detection_and_diagnosis.xlsx`](./results/results_parsed_detection_and_diagnosis.xlsx): compiled results and statistics for CRANE-LLM performance on crash detection and diagnosis
    - [`results_parsed_detection_only.xlsx`](./results/results_parsed_detection_only.xlsx): compiled results and statistics for CRANE-LLM performance, ablation study, and API documentation grounding study on crash detection
    - [`cohens_kappa_human_validation.txt`](./results/cohens_kappa_human_validation.txt): statistics of human evaluation on crash diagnosis
    - [`runtime_doc_token_analysis.txt`](./results/runtime_doc_token_analysis.txt): statistics of tokens of additional API documentation information
    - [`runtime_recording/`](./results/runtime_recording/): statistics of runtime for prior cell executions and querying CRANE-LLM with GPT-5


## License

This project is licensed under the terms of the BSD 3-Clause License.
