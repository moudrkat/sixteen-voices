#!/usr/bin/env python3
"""Probe Q1 (Kill a Head) and Q4 (Blend) demos across varied prompts.

Runs sequentially (CPU). Prints side-by-side outputs so we can judge
whether the effects in the poster app hold up outside the hardcoded
"It was a dark and stormy" prompt.

Usage:
    python demos/probe_poster_robustness.py [--q1-only | --q4-only]

Output is printed to stdout and also saved to
outputs/probe_poster_robustness.txt.
"""

import argparse
import copy
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from sixteen_voices.model import (
    load_adapted_model,
    load_tokenizer,
    get_attn_out,
    get_attn_module,
)
from sixteen_voices.adapter import load_adapter_deltas, delta_to_AB
from sixteen_voices.steering import generate as steer_generate

ADAPTERS_DIR = Path("outputs/authors")
OUT_PATH = Path("outputs/probe_poster_robustness.txt")
SEED = 42
MAX_NEW = 70

PROMPTS = [
    "It was a dark and stormy",   # poster default
    "There was once a little",    # fairy-tale cue
    "The old man sat down and",   # everyday scene
    "She laughed and said",       # dialogue cue
    "The king had three",         # folk structure
]

Q1_AUTHORS = {
    "carroll": {"label": "Carroll", "dominant": 11, "control": 14},
    "poe":     {"label": "Poe",     "dominant": 14, "control": 11},
}

BLEND_A = "carroll"
BLEND_B = "poet"
BLEND_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _banner(buf, text, ch="═"):
    line = ch * 70
    buf.append(f"\n{line}\n{text}\n{line}")


def _gen_killed(model, tokenizer, prompt, head_to_kill):
    scales = {head_to_kill: 0.0} if head_to_kill is not None else None
    attn_out = get_attn_out(model) if scales else None
    return steer_generate(
        model, tokenizer, prompt,
        head_scales=scales, attn_out=attn_out,
        seed=SEED, max_new_tokens=MAX_NEW,
    )


def probe_q1(buf):
    _banner(buf, "Q1 · KILL A HEAD — robustness across prompts", "═")
    tokenizer = load_tokenizer()

    for author_key, info in Q1_AUTHORS.items():
        label = info["label"]
        dom = info["dominant"]
        ctrl = info["control"]

        print(f"\n[loading {label}]", flush=True)
        model = load_adapted_model(str(ADAPTERS_DIR / author_key / "adapter"))

        _banner(buf, f"Author: {label}   dominant=H{dom}   control=H{ctrl}", "─")

        for prompt in PROMPTS:
            print(f"  • {label} | {prompt!r}", flush=True)
            base = _gen_killed(model, tokenizer, prompt, None)
            kill_dom = _gen_killed(model, tokenizer, prompt, dom)
            kill_ctrl = _gen_killed(model, tokenizer, prompt, ctrl)

            buf.append(f"\nPrompt: {prompt!r}")
            buf.append(f"  baseline (all heads on):")
            buf.append(f"    → {base}")
            buf.append(f"  kill H{dom} (DOMINANT — should wreck voice):")
            buf.append(f"    → {kill_dom}")
            buf.append(f"  kill H{ctrl} (control — should barely matter):")
            buf.append(f"    → {kill_ctrl}")

        del model


def _inject(model, deltas):
    attn = get_attn_module(model)
    for proj in ["q_proj", "v_proj"]:
        A, B = delta_to_AB(deltas[proj])
        getattr(attn, proj).lora_A["default"].weight.data.copy_(A)
        getattr(attn, proj).lora_B["default"].weight.data.copy_(B)


def _interp(d_a, d_b, alpha):
    return {
        proj: (1 - alpha) * d_a[proj] + alpha * d_b[proj]
        for proj in ["q_proj", "v_proj"]
    }


def probe_q4(buf):
    _banner(buf, f"Q4 · BLEND {BLEND_A}→{BLEND_B} — robustness across prompts", "═")
    tokenizer = load_tokenizer()

    print("[loading template + deltas]", flush=True)
    template = load_adapted_model(str(ADAPTERS_DIR / BLEND_A / "adapter"))
    d_a = load_adapter_deltas(str(ADAPTERS_DIR / BLEND_A / "adapter"))
    d_b = load_adapter_deltas(str(ADAPTERS_DIR / BLEND_B / "adapter"))

    for prompt in PROMPTS:
        _banner(buf, f"Prompt: {prompt!r}", "─")
        for alpha in BLEND_ALPHAS:
            print(f"  • α={alpha:.2f} | {prompt!r}", flush=True)
            blended = _interp(d_a, d_b, alpha)
            model = copy.deepcopy(template)
            _inject(model, blended)
            text = steer_generate(
                model, tokenizer, prompt,
                seed=SEED, max_new_tokens=MAX_NEW,
            )
            del model
            buf.append(f"  α={alpha:.2f}:")
            buf.append(f"    → {text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q1-only", action="store_true")
    ap.add_argument("--q4-only", action="store_true")
    args = ap.parse_args()

    buf = []
    buf.append(f"Probe run  seed={SEED}  max_new={MAX_NEW}")
    buf.append(f"Prompts ({len(PROMPTS)}): " + "; ".join(repr(p) for p in PROMPTS))

    if not args.q4_only:
        probe_q1(buf)
    if not args.q1_only:
        probe_q4(buf)

    out = "\n".join(buf)
    print(out)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(out)
    print(f"\n[saved to {OUT_PATH}]")


if __name__ == "__main__":
    main()