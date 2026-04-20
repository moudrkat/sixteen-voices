#!/usr/bin/env python3
"""Quick probe: does Poe→Poet blending work as smoothly as Carroll→Poet?"""

import copy
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from sixteen_voices.model import load_adapted_model, load_tokenizer, get_attn_module
from sixteen_voices.adapter import load_adapter_deltas, delta_to_AB
from sixteen_voices.steering import generate as steer_generate

ADAPTERS_DIR = Path("outputs/authors")
OUT_PATH = Path("outputs/probe_poe_poet.txt")
SEED = 42
MAX_NEW = 70

PROMPTS = [
    "It was a dark and stormy",
    "There was once a little",
    "The old man sat down and",
]

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def inject(model, deltas):
    attn = get_attn_module(model)
    for proj in ["q_proj", "v_proj"]:
        A, B = delta_to_AB(deltas[proj])
        getattr(attn, proj).lora_A["default"].weight.data.copy_(A)
        getattr(attn, proj).lora_B["default"].weight.data.copy_(B)


def interp(d_a, d_b, alpha):
    return {p: (1 - alpha) * d_a[p] + alpha * d_b[p]
            for p in ["q_proj", "v_proj"]}


def main():
    tokenizer = load_tokenizer()
    print("[loading template + deltas]", flush=True)
    template = load_adapted_model(str(ADAPTERS_DIR / "poe" / "adapter"))
    d_a = load_adapter_deltas(str(ADAPTERS_DIR / "poe" / "adapter"))
    d_b = load_adapter_deltas(str(ADAPTERS_DIR / "poet" / "adapter"))

    buf = [f"Poe → Poet blend  seed={SEED}  max_new={MAX_NEW}"]
    for prompt in PROMPTS:
        buf.append("\n" + "─" * 70)
        buf.append(f"Prompt: {prompt!r}")
        buf.append("─" * 70)
        for alpha in ALPHAS:
            print(f"  • α={alpha:.2f} | {prompt!r}", flush=True)
            blended = interp(d_a, d_b, alpha)
            model = copy.deepcopy(template)
            inject(model, blended)
            text = steer_generate(model, tokenizer, prompt,
                                  seed=SEED, max_new_tokens=MAX_NEW)
            del model
            buf.append(f"  α={alpha:.2f}:")
            buf.append(f"    → {text}")

    out = "\n".join(buf)
    print(out)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(out)
    print(f"\n[saved to {OUT_PATH}]")


if __name__ == "__main__":
    main()