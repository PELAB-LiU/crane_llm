from ollama import Client
import json
from llms.config_llms import config
import time
import os
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
# from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ast
from transformers import GenerationConfig

load_dotenv()

# define a retry decorator for openai API calls with suggested backoff time
def retry_with_suggested_backoff(func, max_retries: int = 10):
    import openai
    from openai import RateLimitError

    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specific errors
            except Exception as e:
                # Increment retries
                num_retries += 1
                is_rate_limit = False
                if isinstance(e, RateLimitError):
                    is_rate_limit = True
                else:
                    # some wrappers or libs may expose http status; check for 429
                    if hasattr(e, "http_status") and getattr(e, "http_status") == 429:
                        is_rate_limit = True
                    elif "rate limit" in str(e).lower() or "please try again" in str(e).lower():
                        is_rate_limit = True

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                if not is_rate_limit:
                    print(f"LLM call failed with error: {e}. Retrying.")
                else:
                    # Try to extract retry time from common sources
                    delay = None
                    # 1) try headers (if present)
                    headers = getattr(e, "headers", None)
                    if headers:
                        retry_hdr = headers.get("retry-after") or headers.get("Retry-After")
                        if retry_hdr is not None:
                            try:
                                delay = float(retry_hdr)
                            except Exception:
                                pass

                    # 2) try message patterns like "Please try again in 2s" or "Please try again in 1500ms"
                    if delay is None:
                        m = re.search(r"(\d+(?:\.\d+)?)(ms|s|m)\b", str(e), flags=re.IGNORECASE)
                        if m:
                            val = float(m.group(1))
                            unit = m.group(2).lower()
                            if unit == "ms":
                                delay = val / 1000.0
                            elif unit == "s":
                                delay = val
                            elif unit == "m":
                                delay = val * 60.0

                    # 3) fallback to exponential backoff capped to 60s
                    if delay is None:
                        delay = min(2 ** num_retries, 60)

                    # add a small buffer
                    delay = float(delay) + 1.0
                    print(f"Rate limit encountered: retrying in {delay:.1f}s (attempt {num_retries}/{max_retries})")
                    time.sleep(delay)
    return wrapper

class LLMExecutor:
    def __init__(self, model: str = None, libname: str = None, filename: str = None, user_message: str = None, runs: int = 5, MAX_RETRIES: int = 5, RETRY_DELAY: int = 10):
        self._model = model
        self._runs = runs
        self._MAX_RETRIES = MAX_RETRIES
        self._RETRY_DELAY = RETRY_DELAY
        self._filename = filename
        if filename is not None and libname is not None:
            self.user_message = None
            filepath = config.path_input.joinpath(libname).joinpath(filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    self.user_message = f.read()
                except Exception:
                    print(f"Warning: Failed to parse {filename}")
        else:
            if user_message is None:
                print("Warning: Both filename and user_message are None!")
            self.user_message = user_message
        self.predictions = []

    def llm_run_gemini(self):
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=self._model, 
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                temperature=config.param_temperature,
                system_instruction=config.prompt_instruct,
                response_mime_type="application/json",  # force JSON
                response_schema={
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "detection": {"type": "boolean"},
                    },
                    "required": ["detection"]
                }
            ),
            contents=self.user_message,
        )
        print(response.text)

    # run everything at once 
    def llm_multiple_rounds_gemini(self):
        if self.user_message is None:
            print(f"Failed to predict for {self._filename} because the extracted prompt is None")
            return
        output_name = os.path.splitext(self._filename)[0]
        output_file = f"{config.path_res}/{output_name}.json"

        print_head_str = f"Predicting {output_name}..."
        print(print_head_str)

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
                print(print_head_str + f"Attempt {attempt}...")
                response = client.models.generate_content(
                    model=self._model, 
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                        temperature=config.param_temperature,
                        system_instruction=config.prompt_instruct,
                        response_mime_type="application/json",  # force JSON
                        response_schema={
                            "type": "object",
                            "properties": {
                                "reasoning": {"type": "string"},
                                "detection": {"type": "boolean"},
                            },
                            "required": ["detection"]
                        }
                    ),
                    contents=self.user_message,
                )
                content = response.text
                self.predictions = content #.strip()
                print(print_head_str + f"Attempt {attempt} succeed.")
                break
            except Exception as e:
                print(print_head_str + f"Attempt {attempt}: LLM call failed with error: {e}")
                delay = None
                # Try to extract retryDelay if this is a RESOURCE_EXHAUSTED error
                try:
                    err_obj = e.args[0] if hasattr(e, 'args') and len(e.args) > 0 else None
                    arg_str = e.args[0]
                    # Find the first '{' and parse from there
                    idx = arg_str.find('{')
                    if idx != -1:
                        json_part = arg_str[idx:]
                        try:
                            err_obj = ast.literal_eval(json_part)
                        except Exception:
                            err_obj = None
                    elif isinstance(arg_str, dict):
                        err_obj = arg_str

                    if isinstance(err_obj, dict) and err_obj.get('error', {}).get('code') == 429:
                        for detail in err_obj.get('error', {}).get('details', []):
                            if detail.get('@type', '').endswith('RetryInfo'):
                                retry_delay = detail.get('retryDelay')
                                if retry_delay:
                                    m = re.match(r"([\d\.]+)(ms|s|m)", retry_delay)
                                    if m:
                                        val, unit = float(m.group(1)), m.group(2)
                                        delay = {'ms': val / 1000.0, 's': val, 'm': val * 60.0}[unit]
                                break
                except Exception as e:
                    print(f"Failed to extract retry delay: {e}. Fallback to exponential backoff")
                    pass
                # Fallback to exponential backoff
                if delay is None:
                    delay = self._RETRY_DELAY * attempt
                if attempt < self._MAX_RETRIES:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay+1)
                else:
                    print(print_head_str + f"Max retries reached. Error-skipping.")
        
        if self.predictions:
            # Load previous predictions if file exists
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    previous_predictions = json.load(f)
            else:
                previous_predictions = []
            # Parse API response if it's a JSON string with Unicode escapes
            if isinstance(self.predictions, str):
                try:
                    clean_text = json.loads(self.predictions)
                except json.JSONDecodeError:
                    clean_text = self.predictions
            else:
                clean_text = self.predictions
            # Append current run
            previous_predictions.append(clean_text)
            # Save updated list
            with open(output_file, "w") as f:
                json.dump(previous_predictions, f, indent=2)
            print(f"Results are saved in {output_file}.")

    def llm_run_huggingface(self, tokenizer, model):
        # Tokenize input
        messages = [
            {"role": "system", "content": config.prompt_instruct_enforcejson},
            {"role": "user", "content": self.user_message}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = inputs['input_ids']

        gen_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=config.param_temperature,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        # Generate output
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config)

        # Decode and print result
        # The generated tokens after prompt are the ones after input length
        generated_tokens = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # print(response)
        return response.strip()

    def llm_multiple_rounds_huggingface(self, tokenizer, model):
        if self.user_message is None:
            print(f"Failed to predict for {self._filename} because the extracted prompt is None")
            return
        output_name = os.path.splitext(self._filename)[0]
        output_file = f"{config.path_res}/{output_name}.json"

        print_head_str = f"Predicting {output_name}..."
        print(print_head_str)

        # Tokenize input
        messages = [
            {"role": "system", "content": config.prompt_instruct_enforcejson},
            {"role": "user", "content": self.user_message}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = inputs['input_ids']
        gen_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=config.param_temperature,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, generation_config=gen_config)
                generated_tokens = outputs[0][input_ids.shape[-1]:]
                content = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                self.predictions = content.strip()
                print(print_head_str + f"Attempt {attempt} succeed.")
                break
            except Exception as e:
                print(print_head_str + f"Attempt {attempt}: LLM call failed with error: {e}")
                if attempt < self._MAX_RETRIES:
                    print(f"Retrying in {self._RETRY_DELAY*attempt} seconds...")
                    time.sleep(self._RETRY_DELAY*attempt)
                else:
                    # print(f"Max retries reached. Skipping {out_name}.")
                    # return
                    print(print_head_str + f"Max retries reached. Error-skipping.")
        
        if self.predictions:
            # Load previous predictions if file exists
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    previous_predictions = json.load(f)
            else:
                previous_predictions = []
            # Parse API response if it's a JSON string with Unicode escapes
            if isinstance(self.predictions, str):
                try:
                    clean_text = json.loads(self.predictions)
                except json.JSONDecodeError:
                    clean_text = self.predictions
            else:
                clean_text = self.predictions
            # Append current run
            previous_predictions.append(clean_text)
            # Save updated list
            with open(output_file, "w") as f:
                json.dump(previous_predictions, f, indent=2)
            print(f"Results are saved in {output_file}.")


    @retry_with_suggested_backoff
    def _llm_run_openai(self, client):
        response = client.responses.create(
            model = self._model, #"gpt-5",
            input = [
                {
                    "role": "system",
                    "content": config.prompt_instruct_enforcejson
                },
                {
                    "role": "user",
                    "content": self.user_message
                }
            ],
            reasoning={
                "effort": "minimal"
            },
            text={
                "verbosity": "low"
            },
            # temperature = config.param_temperature, # not supported for gpt-5/o1/o1-mini etc
            max_output_tokens = 2048,
            # top_p = 1,
            # frequency_penalty = 0,
            # presence_penalty = 0
        )
        return response.output_text

    def llm_run_openai(self):
        import openai
        from openai import OpenAI
        import tokenize

        client = OpenAI()
        res = self._llm_run_openai(client)
        print(res)

    # run everything at once 
    def llm_multiple_rounds_openai(self):
        import openai
        from openai import OpenAI
        import tokenize

        client = OpenAI()

        if self.user_message is None:
            print(f"Failed to predict for {self._filename} because the extracted prompt is None")
            return
        output_name = os.path.splitext(self._filename)[0]
        output_file = f"{config.path_res}/{output_name}.json"

        print_head_str = f"Predicting {output_name}..."
        print(print_head_str)

        self.predictions = self._llm_run_openai(client)
        
        if self.predictions:
            # Load previous predictions if file exists
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    previous_predictions = json.load(f)
            else:
                previous_predictions = []
            # Parse API response if it's a JSON string with Unicode escapes
            if isinstance(self.predictions, str):
                try:
                    clean_text = json.loads(self.predictions)
                except json.JSONDecodeError:
                    clean_text = self.predictions
            else:
                clean_text = self.predictions
            # Append current run
            previous_predictions.append(clean_text)
            # Save updated list
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(previous_predictions, f, indent=2, ensure_ascii=False)
            print(f"Results are saved in {output_file}.")

    def llm_run_bserver(self):
        client = Client(host='10.129.20.4:9090')
        response = client.chat(
            model=self._model, 
            messages=[
                {"role": "system", "content": config.prompt_instruct},
                {"role": "user", "content": self.user_message}
            ],
            options=config.param_options
        )
        content = response['message']['content']
        print(f"Predition succeed: {content.strip()}")

    def llm_multiple_runs_bserver(self):
        if self.user_message is None:
            print(f"Failed to predict for {self._filename} because the extracted prompt is None")
            return
        output_name = os.path.splitext(self._filename)[0]
        output_file = f"{config.path_res}/{output_name}.json"
        for i in range(self._runs):
            print_head_str = f"Predicting {output_name}: {i}th run..."
            print(print_head_str)

            for attempt in range(1, self._MAX_RETRIES + 1):
                try:
                    client = Client(host='10.129.20.4:9090')
                    print(print_head_str + f"Attempt {attempt}...")
                    response = client.chat(
                        model=self._model, 
                        messages=[
                            {"role": "system", "content": config.prompt_instruct},
                            {"role": "user", "content": self.user_message}
                        ],
                        options=config.param_options
                    )
                    content = response['message']['content']
                    self.predictions.append(content.strip())
                    print(print_head_str + f"Attempt {attempt} succeed.")
                    break
                except Exception as e:
                    print(print_head_str + f"Attempt {attempt}: LLM call failed with error: {e}")
                    if attempt < self._MAX_RETRIES:
                        print(f"Retrying in {self._RETRY_DELAY*attempt} seconds...")
                        time.sleep(self._RETRY_DELAY*attempt)
                    else:
                        # print(f"Max retries reached. Skipping {out_name}.")
                        # return
                        print(print_head_str + f"Max retries reached. Attempt {attempt} written as: [Error-skipping].")
                        self.predictions.append("[Error-skipping]")

        # Save all outputs
        with open(output_file, "w") as f:
            json.dump(self.predictions, f, indent=2)

        print(f"Predictions by {self._model} finished {self._runs} runs, the results are saved in {output_file}.")
