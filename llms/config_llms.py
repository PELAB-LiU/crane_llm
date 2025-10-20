from pathlib import Path

class Config:
    def __init__(self):
        # current configuration setup ---- 

        # [crash detection / crash repair] with [executed code cells / executed code cells and runinfo / runinfo]
        # self.current_task = "crash detection with executed code cells"
        # self.current_task = "crash detection with executed code cells and runinfo"
        self.current_task = "result parsing llm" # result parsing sa
        self.current_llm_model = "gpt-5"
        # [crash detection/localization]
        self._task_abbr = {
            "crash detection with executed code cells": "_code",
            "crash detection with executed code cells and runinfo": "_runinfo",
            "result parsing llm": "", # no use
            "result parsing sa": "", # no use
        }
        self._llm_model_abbr = {
            "gpt-5": "_gpt5",
            "gemini-2.5-flash": "_gemini",
            "Qwen/Qwen2.5-32B-Instruct": "_qwen",
        }

        self.param_temperature = 0.01
        self.param_temperature_result_parsing = 0.01

        self.path_nbs = Path("target_nbs")
        self.path_ground_truth = Path("llms/llms_outputs/ground_truth.xlsx")

        # config define --------
        self.path_input_executed_code = Path("llms/llms_inputs/executed_code")
        self.path_input_executed_code_runinfo = Path("llms/llms_inputs/executed_code_runinfo")
        self.path_input_runinfo = Path("llms/llms_inputs/runinfo")

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

        self.prompt_instruct_crash_detection_2 = ""

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
        self.path_res_parsed_res_llm = Path("llms/llms_outputs/results_parsed_reversed")
        self.path_input_parsed_res_sa = Path("sas/sas_outputs/results_raw/")
        self.path_res_parsed_res_sa = Path("sas/sas_outputs/results_parsed_reversed/")

        # [crash repair]
        self.path_res_crash_repair = Path("llms/llms_outputs/crash_repair")
        self.prompt_instruct_crash_repair_0 = """..."""
        self.prompt_instruct_crash_repair_1 = """..."""
        self.prompt_instruct_crash_repair_2 = """..."""

        self.prompt_instruct_config = {
            "crash detection with executed code cells": self.prompt_instruct_crash_detection_0,
            "crash detection with executed code cells and runinfo": self.prompt_instruct_crash_detection_1,
            "crash detection with runinfo": self.prompt_instruct_crash_detection_2,
            "crash repair with executed code cells": self.prompt_instruct_crash_repair_0,
            "crash repair with executed code cells and runinfo": self.prompt_instruct_crash_repair_1,
            "crash repair with runinfo": self.prompt_instruct_crash_repair_2,
            "result parsing llm": self.prompt_instruct_result_evaluation_llm,
            "result parsing sa": self.prompt_instruct_result_evaluation_sa,
        }

        self.path_config = {
            "crash detection with executed code cells": (lambda: self.path_input_executed_code, lambda: self.path_res_crash_detection),
            "crash detection with executed code cells and runinfo": (lambda: self.path_input_executed_code_runinfo, lambda: self.path_res_crash_detection),
            "crash detection with runinfo": (lambda: self.path_input_runinfo, lambda: self.path_res_crash_detection),
            "crash repair with executed code cells": (lambda: self.path_input_executed_code, lambda: self.path_res_crash_repair),
            "crash repair with executed code cells and runinfo": (lambda: self.path_input_executed_code_runinfo, lambda: self.path_res_crash_repair),
            "crash repair with runinfo": (lambda: self.path_input_runinfo, lambda: self.path_res_crash_repair),
            "result parsing llm": (lambda: self.path_input_parsed_res_llm, lambda: self.path_res_parsed_res_llm),
            "result parsing sa": (lambda: self.path_input_parsed_res_sa, lambda: self.path_res_parsed_res_sa),
        }


    @property
    def path_res_crash_detection(self):
        return Path("llms/llms_outputs/crash_detection"+self._llm_model_abbr[self.current_llm_model]+self._task_abbr[self.current_task])

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
        return self.path_config[self.current_task][0]()

    @property
    def path_res(self):
        return self.path_config[self.current_task][1]()

config = Config()