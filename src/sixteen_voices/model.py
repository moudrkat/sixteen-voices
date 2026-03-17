"""Model loading utilities."""

from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

from .constants import MODEL_NAME, RANK, LORA_ALPHA, TARGET_MODULES


def load_tokenizer():
    """Load tokenizer with pad_token set."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model():
    """Load the base TinyStories model (no adapter)."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    return model


def load_adapted_model(adapter_path: str | Path):
    """Load base model + LoRA adapter. Returns (model, base_ref)."""
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model


def create_lora_model(base_model=None, rank=RANK, alpha=LORA_ALPHA):
    """Wrap a base model with a fresh LoRA config. Returns PeftModel.

    If base_model is provided, it is deep-copied so the original stays clean.
    """
    import copy
    if base_model is None:
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    else:
        base_model = copy.deepcopy(base_model)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, config)


def get_attn_module(model):
    """Return the single attention layer's attention module."""
    # LoRA-wrapped models have base_model.model; plain models don't
    if hasattr(model, "peft_config"):
        return model.base_model.model.transformer.h[0].attn.attention
    return model.transformer.h[0].attn.attention


def get_attn_out(model):
    """Return the output projection of the attention layer (for hooks)."""
    return get_attn_module(model).out_proj
