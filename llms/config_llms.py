from pathlib import Path

class Config:
    def __init__(self):
        # current configuration setup ---- 

        # [crash detection] with [executed code cells / executed code cells and runinfo / runinfo]
        # self.current_task = "crash detection with executed code cells"
        # self.current_task = "crash detection with executed code cells and runinfo"
        self.current_task = "result parsing llm" # result parsing llm diagnosis only, result parsing sa
        self.current_llm_model = "gpt-5"
        self.current_ablation_setting = "full" # full / r_v / s_r / s_v
        self.current_doc = False # True / False
        # [crash detection/localization]
        self._task_abbr = {
            "crash detection with executed code cells": "_code",
            "crash detection with executed code cells and runinfo": "_runinfo",
            "result parsing llm": "", # no use
            "result parsing llm diagnosis only": "", # no use
            "result parsing sa": "", # no use
        }
        self._llm_model_abbr = {
            "gpt-5": "_gpt5",
            "gemini-2.5-flash": "_gemini",
            "Qwen/Qwen2.5-Coder-32B-Instruct": "_qwen",
            "gpt-5-mini": "_gpt5mini",
            "Qwen/QwQ-32B": "_qwq32b",
            "AtlaAI/Selene-1-Mini-Llama-3.1-8B": "_llama8b"
        }

        self.param_temperature = 0.01
        self.param_temperature_result_parsing = 0.01

        self.path_nbs = Path("target_nbs")
        self.path_ground_truth = Path("llms/llms_outputs/ground_truth_crash_prediction.xlsx")

        # config define --------
        self.path_input_executed_code = Path("llms/llms_inputs/executed_code")
        self.path_input_executed_code_runinfo = Path("llms/llms_inputs/executed_code_runinfo_full")
        self.path_input_executed_code_runinfo_r_v = Path("llms/llms_inputs/executed_code_runinfo_r_v")
        self.path_input_executed_code_runinfo_s_r = Path("llms/llms_inputs/executed_code_runinfo_s_r")
        self.path_input_executed_code_runinfo_s_v = Path("llms/llms_inputs/executed_code_runinfo_s_v")
        self.path_input_executed_code_runinfo_doc = Path("llms/llms_inputs/executed_code_runinfo_full_doc")

        # cot
        # not enforce json output because API supports JSON format outputs
        self.prompt_instruct_crash_detection_0 = """You are an automated crash detector for ML notebooks.
Given:
- a set of [Executed Cells] that have already run successfully,
- a [Target Cell] that may or may not crash when executed,

Your task is to reason step by step whether executing the [Target Cell] will crash.

Output:
- reasoning: a short explanation (concise: 1-2 sentences),
- detection: `true` only if you are CERTAIN the [Target Cell] will crash, otherwise output `false`.a

--- Input Begins Below ---
        """

        self.prompt_instruct_crash_detection_1 = """You are an automated crash detector for ML notebooks.
Given:
- a set of [Executed Cells] that have already run successfully,
- a [Target Cell] that may or may not crash when executed,
- and additional [Current relevant runtime information] such as variable values or types that are relevent to the [Target Cell],

Your task is to reason step by step whether executing the [Target Cell] will crash.

Output:
- reasoning: a short explanation (concise: 1-2 sentences),
- detection: `true` only if you are CERTAIN the [Target Cell] will crash, otherwise output `false`.

--- Input Begins Below ---
        """
        # enforce json format output in prompt
        self.prompt_instruct_crash_detection_0_enforcejson = """You are an automated crash detector for ML notebooks.
Given:
- a set of [Executed Cells] that have already run successfully,
- a [Target Cell] that may or may not crash when executed,

Your task is to reason step by step whether executing the [Target Cell] will crash.

Important output rules:
- Output EXACTLY one JSON object and NOTHING else. Use JSON booleans true and false (lowercase).
- Schema:
  {
    "reasoning": string,
    "detection": boolean
  }
- Give a short explanation in `reasoning` (concise: 1-2 sentences).
- If you are not CERTAIN the cell will crash, you MUST output `"detection": false`.
- Your output must ONLY be the JSON object.

--- Input Begins Below ---
        """

        self.prompt_instruct_crash_detection_1_enforcejson = """You are an automated crash detector for ML notebooks.
Given:
- a set of [Executed Cells] that have already run successfully,
- a [Target Cell] that may or may not crash when executed,
- and additional [Current relevant runtime information] such as variable values or types that are relevent to the [Target Cell],

Your task is to reason step by step whether executing the [Target Cell] will crash.

Important output rules:
- Output EXACTLY one JSON object and NOTHING else. Use JSON booleans true and false (lowercase).
- Schema:
  {
    "reasoning": string,
    "detection": boolean
  }
- Give a short explanation in `reasoning` (concise: 1-2 sentences).
- If you are not CERTAIN the cell will crash, you MUST output `"detection": false`.
- Your output must ONLY be the JSON object.

--- Input Begins Below ---
        """

        # [result parsing]
        # not trust LLM prediction labels
        self.prompt_instruct_result_evaluation_llm = """
You are a judge evaluating a crash prediction from an LLM.
Your output must be EXACTLY one of: [Correct], [Partially correct], [Reasoning Wrong], [Wrong].

Inputs:
- LLM detection (true/false) with reasons;
- Ground truth detection (true = crash, false = no crash) with reasons (if true).

Evaluation Rules:
1. Compare detection labels first:
    - If LLM = false and ground truth = false → [Correct];
    - If LLM = false and ground truth = true → [Wrong];
    - If LLM = true and ground truth = false → [Wrong];
    - If LLM = true and ground truth = true -> go to step 2:
2. Validate reasoning (only if both are true):
    - If every LLM reason aligns with one or more ground truth reasons → [Correct];
    - If some but not all LLM reasons align with ground truth reasons → [Partially correct];
    - If none of the LLM reasons align with the ground truth reasons → [Reasoning Wrong].

Final Answer: Output ONLY one of [Correct], [Partially correct], [Reasoning Wrong], [Wrong].
        """

        self.prompt_instruct_result_evaluation_llm_diagnosis_only = """
        You are a judge evaluating a diagnosis of a predicted code crash from an LLM.
Your output must be EXACTLY one of: [Correct], [Partially correct], [Reasoning Wrong].

Inputs:
    - LLM predicted code crashing reasons;
    - Ground truth code crashing reasons.

Evaluation Rules:
    - If every LLM reason aligns with one or more ground truth reasons → [Correct];
    - If some but not all LLM reasons align with ground truth reasons → [Partially correct];
    - If none of the LLM reasons align with the ground truth reasons → [Reasoning Wrong].

Final Answer: Output ONLY one of [Correct], [Partially correct], [Reasoning Wrong].
        """

#         self.prompt_instruct_result_evaluation_llm_diagnosis_only = """
#         You are a judge evaluating a diagnosis of a predicted code crash from an LLM.
# Your output must be EXACTLY one of: [Correct], [Reasoning Wrong].

# Inputs:
#     - LLM predicted code crashing reasons;
#     - Ground truth code crashing reasons.

# Evaluation Rules:
#     - If every LLM reason aligns with one or more ground truth reasons → [Correct];
#     - If only some or none of the LLM reasons align with the ground truth reasons → [Reasoning Wrong].

# Final Answer: Output ONLY one of [Correct], [Reasoning Wrong].
#         """

        self.prompt_instruct_result_evaluation_sa = """
You are a judge evaluating a crash prediction reported by a static analyzer (such as pylint or pyright). 
Your output must be EXACTLY one of: [Correct], [Partially correct], [Reasoning Wrong], [Wrong].

Inputs:
- Ground truth detection (true = crash, false = no crash) with reasons (if true);
- Static analyzer prediction (non-empty/list of errors = crash, empty = no crash).

Evaluation Rules:
1. Check detection result first:
    - If analyzer prediction is empty and ground truth = false → [Correct];
    - If analyzer prediction is empty and ground truth = true → [Wrong];
    - If analyzer prediction is non-empty and ground truth = false → [Wrong];
    - If analyzer prediction is non-empty and ground truth = true -> go to step 2:
2. Validate reasoning (only if both predict crash):
    - If every reported errors from the analyzer aligns with one or more ground truth reasons → [Correct];
    - If some but not all reported errors align with ground truth reasons → [Partially correct];
    - If none of the reported errors align with the ground truth reasons → [Reasoning Wrong].

Final Answer: Output ONLY one of [Correct], [Partially correct], [Reasoning Wrong], [Wrong].
"""

        self.path_input_parsed_res_llm = Path("llms/llms_outputs/results_raw")
        self.path_res_parsed_res_llm = Path("llms/llms_outputs/results_parsed")
        self.path_input_parsed_res_sa = Path("sas/sas_outputs/results_raw/")
        self.path_res_parsed_res_sa = Path("sas/sas_outputs/results_parsed/")


        self.prompt_instruct_config = {
            "crash detection with executed code cells": self.prompt_instruct_crash_detection_0,
            "crash detection with executed code cells and runinfo": self.prompt_instruct_crash_detection_1,
            "result parsing llm": self.prompt_instruct_result_evaluation_llm,
            "result parsing llm diagnosis only": self.prompt_instruct_result_evaluation_llm_diagnosis_only,
            "result parsing sa": self.prompt_instruct_result_evaluation_sa,
        }

        self.path_config = {
            "crash detection with executed code cells": (lambda: self.path_input_executed_code, lambda: self.path_res_crash_detection),
            "crash detection with executed code cells and runinfo(full)": (lambda: self.path_input_executed_code_runinfo, lambda: self.path_res_crash_detection),
            "crash detection with executed code cells and runinfo(r_v)": (lambda: self.path_input_executed_code_runinfo_r_v, lambda: self.path_res_crash_detection),
            "crash detection with executed code cells and runinfo(s_r)": (lambda: self.path_input_executed_code_runinfo_s_r, lambda: self.path_res_crash_detection),
            "crash detection with executed code cells and runinfo(s_v)": (lambda: self.path_input_executed_code_runinfo_s_v, lambda: self.path_res_crash_detection),
            "result parsing llm": (lambda: self.path_input_parsed_res_llm, lambda: self.path_res_parsed_res_llm),
            "result parsing llm diagnosis only": (lambda: self.path_input_parsed_res_llm, lambda: self.path_res_parsed_res_llm),
            "result parsing sa": (lambda: self.path_input_parsed_res_sa, lambda: self.path_res_parsed_res_sa),
        }


    @property
    def path_res_crash_detection(self):
        output_path = "llms/llms_outputs/crash_detection"+self._llm_model_abbr[self.current_llm_model]+self._task_abbr[self.current_task]
        if "runinfo" in self.current_task:
            if self.current_doc:
                return Path(output_path+"_full_doc")
            return Path(output_path+"_"+self.current_ablation_setting)
        return Path(output_path)

    @property
    def param_options(self):
        if "result parsing" in self.current_task:
            return {"temperature": self.param_temperature_result_parsing}
        else:
            return {"temperature": self.param_temperature} #, "max_tokens": 128000

    @property
    def prompt_instruct(self):
        return self.prompt_instruct_config[self.current_task]

    @property
    def prompt_instruct_enforcejson(self):
        if self.current_task == "crash detection with executed code cells":
            return self.prompt_instruct_crash_detection_0_enforcejson
        elif self.current_task == "crash detection with executed code cells and runinfo":
            return self.prompt_instruct_crash_detection_1_enforcejson
        else:
            return self.prompt_instruct_config[self.current_task]

    @property
    def path_input(self):
        if "runinfo" in self.current_task:
            if self.current_doc:
                return self.path_input_executed_code_runinfo_doc
            return self.path_config[self.current_task+"({})".format(self.current_ablation_setting)][0]()
        return self.path_config[self.current_task][0]()

    @property
    def path_res(self):
        if "runinfo" in self.current_task:
            return self.path_config[self.current_task+"({})".format(self.current_ablation_setting)][1]()
        return self.path_config[self.current_task][1]()

    def get_model_abbr(self, model_name):
        return self._llm_model_abbr.get(model_name, "_unknown")
        
config = Config()