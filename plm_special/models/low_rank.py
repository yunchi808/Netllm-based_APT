"""
LoRA (PEFT) support for APT PLM fine-tuning.
Mirrors ABR's low_rank module so APT can use larger PLMs with rank > 0.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType

# Module names per architecture for LoRA target_modules
TARGET_MODULES = {
    "llama": ["q_proj", "v_proj"],
    "llava": ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "t5-lm": ["q", "v"],
}


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def peft_model(plm, plm_type: str, rank: int, print_trainable=False, task_type=TaskType.FEATURE_EXTRACTION):
    for param in plm.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    if hasattr(plm, "gradient_checkpointing_enable"):
        plm.gradient_checkpointing_enable()
    if hasattr(plm, "enable_input_require_grads"):
        plm.enable_input_require_grads()

    if plm_type not in TARGET_MODULES:
        raise ValueError(f"plm_type must be one of {list(TARGET_MODULES.keys())}, got {plm_type!r}")

    config = LoraConfig(
        r=rank,
        lora_alpha=32,
        target_modules=TARGET_MODULES[plm_type],
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
    )
    model = get_peft_model(plm, config)
    if print_trainable:
        print_trainable_parameters(model)
    return model
