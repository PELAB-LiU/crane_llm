# qwen_loader.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Cache the model and tokenizer so they load only once
_tokenizer = None
_model = None

def get_qwen_model(llm_model: str):
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model  # Already loaded

    # Print CUDA info
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPU(s)")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    #     max_memory = {i: "40GiB" for i in range(num_gpus)} if num_gpus > 0 else None
    # else:
    #     print("No GPUs available!")
    #     max_memory = {"cpu": "64GiB"}

    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)

    # Load model
    _model = AutoModelForCausalLM.from_pretrained(
        llm_model,
        device_map="auto",
        # max_memory=max_memory,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        # offload_folder="offload"
    )

    return _tokenizer, _model
