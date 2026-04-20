#!/usr/bin/env python3
"""Probe whether H6 is truly silent for Carroll/Poe/Grimm.

If yes, (dominant, H6) is a clean pair for the Q1 side-by-side demo:
  - kill dominant → obviously breaks voice
  - kill H6 → text barely changes
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from sixteen_voices.model import (
    load_adapted_model, load_tokenizer, get_attn_out,
)
from sixteen_voices.steering import generate as steer_generate

ADAPTERS_DIR = Path("outputs/authors")
OUT_PATH = Path("outputs/probe_kill_control.txt")
SEED = 42
MAX_NEW = 70

PROMPTS = [
    "It was a dark and stormy",
    "There was once a little",
    "The old man sat down and",
    "She laughed and said",
    "The king had three",
]

PAIRS = {
    "carroll": ("Carroll", 11, 6),
    "poe":     ("Poe",     14, 6),
    "grimm":   ("Grimm",   11, 6),
}


def gen_kill(model, tokenizer, prompt, head):
    scales = {head: 0.0} if head is not None else None
    attn_out = get_attn_out(model) if scales else None
    return steer_generate(
        model, tokenizer, prompt,
        head_scales=scales, attn_out=attn_out,
        seed=SEED, max_new_tokens=MAX_NEW,
    )


def main():
    tokenizer = load_tokenizer()
    buf = [f"H6 silent-control probe  seed={SEED}  max_new={MAX_NEW}"]

    for key, (label, dom, ctrl) in PAIRS.items():
        print(f"[loading {label}]", flush=True)
        model = load_adapted_model(str(ADAPTERS_DIR / key / "adapter"))
        buf.append("\n" + "═" * 70)
        buf.append(f"{label}   dominant=H{dom}   control=H{ctrl}")
        buf.append("═" * 70)
        for prompt in PROMPTS:
            print(f"  • {label} | {prompt!r}", flush=True)
            base = gen_kill(model, tokenizer, prompt, None)
            kd = gen_kill(model, tokenizer, prompt, dom)
            kc = gen_kill(model, tokenizer, prompt, ctrl)
            buf.append(f"\nPrompt: {prompt!r}")
            buf.append(f"  baseline:          {base}")
            buf.append(f"  kill H{dom} (DOM):     {kd}")
            buf.append(f"  kill H{ctrl} (control):  {kc}")
        del model

    out = "\n".join(buf)
    print(out)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(out)
    print(f"\n[saved to {OUT_PATH}]")


if __name__ == "__main__":
    main()