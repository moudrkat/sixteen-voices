"""Attention head steering via forward hooks."""

import torch

from .constants import HEAD_DIM


def make_hook(head_scales: dict[int, float]):
    """Create a forward pre-hook that scales specific heads' output before W_O.

    Args:
        head_scales: {head_index: scale_factor} e.g. {14: 2.0, 3: 0.5}
    """
    def hook_fn(module, args):
        h = args[0].clone()
        for head_idx, scale in head_scales.items():
            s = head_idx * HEAD_DIM
            h[:, :, s : s + HEAD_DIM] *= scale
        return (h,) + args[1:]
    return hook_fn


def generate(
    model,
    tokenizer,
    prompt: str,
    head_scales: dict[int, float] | None = None,
    attn_out=None,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int | None = None,
) -> str:
    """Generate text, optionally with head steering.

    Args:
        model: The language model.
        tokenizer: Tokenizer.
        prompt: Input prompt string.
        head_scales: Optional head scaling dict for steering.
        attn_out: The attention output projection module (needed if head_scales given).
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        seed: Random seed for reproducibility.

    Returns:
        Generated text (excluding the prompt).
    """
    if seed is not None:
        torch.manual_seed(seed)

    hook = None
    if head_scales and attn_out:
        hook = attn_out.register_forward_pre_hook(make_hook(head_scales))

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()
    finally:
        if hook:
            hook.remove()
    return text


def steered_perplexity(
    model,
    tokenizer,
    eval_text: str,
    head_scales: dict[int, float] | None = None,
    attn_out=None,
    max_length: int = 512,
) -> float:
    """Compute perplexity with optional head steering."""
    hook = None
    if head_scales and attn_out:
        hook = attn_out.register_forward_pre_hook(make_hook(head_scales))

    try:
        inputs = tokenizer(eval_text, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
        val = torch.exp(out.loss).item()
    finally:
        if hook:
            hook.remove()
    return val
