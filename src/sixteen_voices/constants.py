"""Model and experiment constants."""

MODEL_NAME = "roneneldan/TinyStories-1Layer-21M"
NUM_HEADS = 16
HEAD_DIM = 64
HIDDEN_DIM = 1024  # NUM_HEADS * HEAD_DIM
RANK = 8
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "v_proj"]
