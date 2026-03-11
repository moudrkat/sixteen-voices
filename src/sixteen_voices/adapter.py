"""LoRA adapter weight manipulation — loading, knockout, SVD."""

from pathlib import Path

import torch
from safetensors.torch import load_file

from .constants import HEAD_DIM, NUM_HEADS, RANK


_WEIGHT_PREFIX = "base_model.model.transformer.h.0.attn.attention"


def load_adapter_deltas(adapter_path: str | Path) -> dict[str, torch.Tensor]:
    """Load adapter weights and compute delta = B @ A for q_proj and v_proj.

    Returns dict with keys "q_proj", "v_proj", each a [1024, 1024] tensor.
    """
    weights = load_file(str(Path(adapter_path) / "adapter_model.safetensors"))
    deltas = {}
    for proj in ["q_proj", "v_proj"]:
        A = weights[f"{_WEIGHT_PREFIX}.{proj}.lora_A.weight"]
        B = weights[f"{_WEIGHT_PREFIX}.{proj}.lora_B.weight"]
        deltas[proj] = B @ A
    return deltas


def knockout_all_except(delta: torch.Tensor, keep_head: int) -> torch.Tensor:
    """Zero out all head rows except keep_head."""
    delta_ko = torch.zeros_like(delta)
    start = keep_head * HEAD_DIM
    end = start + HEAD_DIM
    delta_ko[start:end, :] = delta[start:end, :]
    return delta_ko


def delta_to_AB(delta: torch.Tensor, rank: int = RANK) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-factorize a delta matrix into LoRA A, B via truncated SVD."""
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    B = U[:, :rank] * S[:rank].unsqueeze(0)
    A = Vh[:rank, :]
    return A, B


def inject_knockout(model, deltas: dict, keep_head: int):
    """Modify a PeftModel in-place so only keep_head's LoRA contribution remains."""
    from .model import get_attn_module

    attn = get_attn_module(model)
    for proj in ["q_proj", "v_proj"]:
        new_delta = knockout_all_except(deltas[proj], keep_head)
        A_new, B_new = delta_to_AB(new_delta)
        getattr(attn, proj).lora_A["default"].weight.data.copy_(A_new)
        getattr(attn, proj).lora_B["default"].weight.data.copy_(B_new)
