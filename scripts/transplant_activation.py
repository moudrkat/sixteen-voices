#!/usr/bin/env python3
"""Activation-based head transplant: swap head outputs at inference time.

Instead of splicing weight rows and doing lossy SVD re-factorization,
this runs both adapters and swaps head activations directly.
No information loss.

Usage:
    python scripts/transplant_activation.py
"""

from pathlib import Path

import torch

from sixteen_voices import (
    generate,
    load_adapted_model,
    load_tokenizer,
)
from sixteen_voices.model import get_attn_module
from sixteen_voices.constants import HEAD_DIM

ADAPTERS_DIR = Path("outputs/authors")
PROMPT = "It was a dark and stormy"
SEED = 42
HEAD = 14


def make_transplant_hook(donor_model, head, head_dim=HEAD_DIM):
    """Hook that replaces one head's output with the donor's cached output.

    We first run the donor model on the same input to cache its head
    activations, then use this hook on the host model to splice them in.
    """
    donor_activations = {}

    def donor_capture_hook(module, args):
        """Capture the donor's pre-W_O head activations."""
        h = args[0]
        s = head * head_dim
        donor_activations["head_out"] = h[:, :, s:s + head_dim].clone()

    def host_replace_hook(module, args):
        """Replace one head in the host's pre-W_O input with the donor's."""
        if "head_out" not in donor_activations:
            return args
        h = args[0].clone()
        s = head * head_dim
        donor_h = donor_activations["head_out"]
        if donor_h.shape[1] <= h.shape[1]:
            h[:, :donor_h.shape[1], s:s + head_dim] = donor_h
        return (h,) + args[1:]

    return donor_capture_hook, host_replace_hook, donor_activations


def generate_with_transplant(host_model, donor_model, tokenizer, prompt,
                             head=HEAD, seed=SEED, max_new_tokens=160):
    """Generate from host model with one head's activations from donor.

    For each token, we need both models to process the same input.
    Since generate() is autoregressive, we do manual token-by-token generation.
    """
    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    plen = input_ids.shape[1]

    host_attn = get_attn_module(host_model)
    donor_attn = get_attn_module(donor_model)

    for _ in range(max_new_tokens):
        # Run donor to capture head activations
        donor_head_out = {}

        def capture_hook(module, args):
            h = args[0]
            s = head * HEAD_DIM
            donor_head_out["val"] = h[:, -1:, s:s + HEAD_DIM].clone()

        hook_d = donor_attn.out_proj.register_forward_pre_hook(capture_hook)
        with torch.no_grad():
            donor_model(input_ids)
        hook_d.remove()

        # Run host with donor's head spliced in
        def replace_hook(module, args):
            h = args[0].clone()
            s = head * HEAD_DIM
            h[:, -1:, s:s + HEAD_DIM] = donor_head_out["val"]
            return (h,) + args[1:]

        hook_h = host_attn.out_proj.register_forward_pre_hook(replace_hook)
        with torch.no_grad():
            logits = host_model(input_ids).logits[:, -1, :]
        hook_h.remove()

        # Sample next token (same as generate())
        logits = logits / 0.8  # temperature
        top_k_logits, top_k_indices = torch.topk(logits, 50)
        probs = torch.softmax(top_k_logits, dim=-1)
        idx = torch.multinomial(probs, 1)
        next_token = top_k_indices.gather(1, idx)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(input_ids[0][plen:], skip_special_tokens=True).strip()
    return text


def main():
    tokenizer = load_tokenizer()

    hosts = ["carroll", "grimm", "minimalist"]
    donor_name = "poe"

    donor_model = load_adapted_model(str(ADAPTERS_DIR / donor_name / "adapter"))

    for host_name in hosts:
        host_path = str(ADAPTERS_DIR / host_name / "adapter")
        host_model = load_adapted_model(host_path)

        # Pure host
        torch.manual_seed(SEED)
        pure = generate(host_model, tokenizer, PROMPT, seed=SEED, max_new_tokens=160)

        # Weight-space transplant (old approach, for comparison)
        from sixteen_voices import load_adapter_deltas
        from sixteen_voices.adapter import delta_to_AB
        r_deltas = load_adapter_deltas(host_path)
        d_deltas = load_adapter_deltas(str(ADAPTERS_DIR / donor_name / "adapter"))
        t_deltas = {}
        for proj in ["q_proj", "v_proj"]:
            result = r_deltas[proj].clone()
            s, e = HEAD * HEAD_DIM, (HEAD + 1) * HEAD_DIM
            result[s:e, :] = d_deltas[proj][s:e, :]
            t_deltas[proj] = result
        weight_model = load_adapted_model(host_path)
        attn = get_attn_module(weight_model)
        for proj in ["q_proj", "v_proj"]:
            A, B = delta_to_AB(t_deltas[proj])
            getattr(attn, proj).lora_A["default"].weight.data.copy_(A)
            getattr(attn, proj).lora_B["default"].weight.data.copy_(B)
        torch.manual_seed(SEED)
        weight_transplant = generate(weight_model, tokenizer, PROMPT, seed=SEED, max_new_tokens=160)
        del weight_model

        # Activation transplant (new approach)
        torch.manual_seed(SEED)
        act_transplant = generate_with_transplant(
            host_model, donor_model, tokenizer, PROMPT,
            head=HEAD, seed=SEED, max_new_tokens=160)

        print(f"\n{'='*80}")
        print(f"  {host_name.upper()} + {donor_name.upper()}'s H{HEAD}")
        print(f"{'='*80}")
        print(f"\n  PURE {host_name.upper()}:")
        print(f"  {pure}\n")
        print(f"  WEIGHT TRANSPLANT (SVD, lossy):")
        print(f"  {weight_transplant}\n")
        print(f"  ACTIVATION TRANSPLANT (exact):")
        print(f"  {act_transplant}\n")

        del host_model


if __name__ == "__main__":
    main()