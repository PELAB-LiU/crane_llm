# CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks

This is the official repository for our paper "CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks". In this paper, we propose CRANE-LLM, a novel approach that augments LLMs with runtime information extracted from the notebook kernel state to enhance their prediction and explanation of ML notebook crashes.

## Repository structure and reproducibility details
Dataset: We use [**Junobench**]((https://huggingface.co/datasets/PELAB-LiU/JunoBench)) dataset in our experiments.

LLMs: LLMs include Gemini (Gemini-2.5-Flash), Qwen (Qwen-2.5-Coder-32B-Instruct), GPT-5.

Repository structure:
- [`config_llms.py`](./llms/config_llms.py): configuration for experiments, including prompts, LLM configs, and related input and output paths
- [`crane-llm.py`](./crane-llm.py): script to run CRANE-LLM given a target notebook
- [`main_LLM.py`](./main_LLM.py): script to run the experiment pipeline, including batch run all notebooks in the dataset for a specific task and experimental setting
- [`main.py`](./main.py): script to run result compilation and analysis
- [`runinfo_parser`](./runinfo_parser): scripts and configuration for *runtime information extraction*
- [`llms`](./llms): LLM-related experiments
    - [`llms_inputs/`](./llms/llms_inputs): generated inputs (executed code cells only, executed code cells with runtime information) to the LLMs
    - [`llms_outputs/`](./llms/llms_outputs): generated outputs by the LLMs
        - [`results_raw/`](./llms/llms_outputs/results_raw/): crash prediction outputs organized into: crane\_detetcion\_[*LLM*]\_[*experimental setup*]\_[*runtime category ablation setup or API grounding*]
        - [`ground_truth_crash_prediction.xlsx`](./llms/llms_outputs/ground_truth_crash_prediction.xlsx): ground truth labels used for evaluating LLM predictions as well as downstream analysis, provided by JunoBench
    - [`prompt_extractor.py`](./llms/prompt_extractor.py): script for constructing prompts (i.e., `llms_inputs/`)
    - [`llm_executor.py`](./llms/llm_executor.py): script for querying LLMs to generate outputs in `llms_outputs/`
- [`results`](./results): results and compiled statistics
    - [`results_parsed_detection_and_diagnosis.xlsx`](./results/results_parsed_detection_and_diagnosis.xlsx): CRANE-LLM performance on the joint crash detection and diagnosis task
        - Sheet "Final_evaluation": Detailed prediction outcomes per LLM per experimental setup. Settings include:
            - code: -RT
            - runinfo: +RT (CRANE-LLM)
        - Sheet "Results_summary": Compiled results and statistics on crash prediction and diagnosis performance of CRANE-LLM
    - [`results_parsed_detection_only.xlsx`](./results/results_parsed_detection_only.xlsx): CRANE-LLM performance on the joint crash detection-only task
        - Sheet "Final_evaluation": Detailed prediction outcomes per LLM per experimental setup including *runtime information category ablation study*, and *API documentation grounding study*. All settings include:
            - code: -RT
            - runinfo: +RT (CRANE-LLM)
            - runinfo_r_v: +RT-S (CRANE-LLM - S), ablated structural runtime information
            - runinfo_s_r: +RT-V (CRANE-LLM - V), ablated value semantics runtime information
            - runinfo_s_v: +RT-R (CRANE-LLM - R), ablated type-level (representation and type semantics) runtime information
            - runinfo_full_doc: +RT+doc (CRANE-LLM + doc), full runtime information with additional API documentation information
        - Sheet "Results_summary": Compiled results and statistics on crash detection performance of CRANE-LLM, including runtime information category ablation study and API documentation grounding study results (and token analysis results)
    - [`runtime_doc_token_analysis.txt`](./results/runtime_doc_token_analysis.txt): statistics of tokens of additional API documentation information, results gained by running script [`token_analysis.py`](./utils/token_analysis.py)
    - [`pairwise_significance_detection_and_diagnosis.json`](./results/pairwise_significance_detection_and_diagnosis.json) and [`pairwise_significance_detection_only.json`](./results/pairwise_significance_detection_only.json): statistical test results of the joint detection and diagnosis task and the detection-only task, the statistics tests are ran by script [statistical_test.py](./utils/statistical_test.py)
    - [`cohens_kappa_human_validation.txt`](./results/cohens_kappa_human_validation.txt): statistics of human evaluation on crash diagnosis
    - [`runtime_recording/`](./results/runtime_recording/): statistics of runtime for prior cell executions and querying CRANE-LLM(GPT-5)

## Environment

To ensure full reproducibility, we provide a docker image (digest: sha256:ecb5753d1cdfc9f0d5dfeb59818cde5be5be2f79541c5facf99761393919e171):
```bash
docker pull yarinamomo/crane_env:latest
```
Then run the docker container:
```bash
docker run -v [volumn_mount_windows_path]:/cranellm_env -w /cranellm_env -p 8888:8888 -it yarinamomo/crane_env:latest /bin/bash
```
Then you can attach this environment to **VS Code** "*Dev Containers: Attach to Running Container...*"

For the commercial LLMs used in the experiments (Gemini and GPT-5), please ensure that the API keys are properly set up before running the scripts (for example, set as global environment variable or config in `.env`). Open-source LLMs (Qwen) can be run directly; however, note that execution may take longer depending on the computational resources available.

## License

This project is licensed under the terms of the BSD 3-Clause License.
