"""Sixteen Voices — Attention head specialization in LoRA-adapted language models."""

from .constants import MODEL_NAME, NUM_HEADS, HEAD_DIM, HIDDEN_DIM, RANK
from .text import extract_prose, compute_perplexity, clean_text
from .model import load_tokenizer, load_base_model, load_adapted_model, create_lora_model, get_attn_out
from .adapter import load_adapter_deltas, knockout_all_except, delta_to_AB, inject_knockout
from .steering import make_hook, generate, steered_perplexity
from .dataset import TextChunkDataset

__all__ = [
    # Constants
    "MODEL_NAME", "NUM_HEADS", "HEAD_DIM", "HIDDEN_DIM", "RANK",
    # Text
    "extract_prose", "compute_perplexity", "clean_text",
    # Model
    "load_tokenizer", "load_base_model", "load_adapted_model", "create_lora_model", "get_attn_out",
    # Adapter
    "load_adapter_deltas", "knockout_all_except", "delta_to_AB", "inject_knockout",
    # Steering
    "make_hook", "generate", "steered_perplexity",
    # Dataset
    "TextChunkDataset",
]
