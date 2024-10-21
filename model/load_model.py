from typing import Dict, Optional, Union

import torch
from langchain.llms import HuggingFacePipeline
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)

_DEFAULT_MODEL_ID = "rtzr/ko-gemma-2-9b-it"


def load_model(
    model_id: Optional[str] = None,
    quantization: bool = True,
    device_map: Union[int, str] = "auto",
    max_new_tokens: int = 450,
    model_kwargs: Optional[Dict] = None,
):
    model_id = model_id or _DEFAULT_MODEL_ID
    model_kwargs = model_kwargs or {}
    bnb_config = None

    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False

    kwargs = {
        "pretrained_model_name_or_path": model_id,
        "device_map": device_map,
        "trust_remote_code": True,
    }

    if bnb_config:
        kwargs.update({"quantization_config": bnb_config})
    
    kwargs.update(model_kwargs)

    model = AutoModelForCausalLM.from_pretrained(**kwargs)

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        return_full_text=False,
        max_new_tokens=max_new_tokens,
    )

    return HuggingFacePipeline(pipeline=text_generation_pipeline)