#!/usr/bin/env python3
"""Probe Experiment A: keep only one head's LoRA delta, check generated text.

For each author, compare:
  - baseline (full adapter, all 16 heads' LoRA active)
  - keep only dominant head's LoRA (should preserve most of voice)
  - keep only low-recovery head's LoRA (should collapse toward base model)
"""

import copy
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from sixteen_voices.model import load_adapted_model, load_tokenizer
from sixteen_voices.adapter import load_adapter_deltas, inject_knockout
from sixteen_voices.steering import generate as steer_generate

ADAPTERS_DIR = Path("outputs/authors")
OUT_PATH = Path("outputs/probe_keep_only.txt")
SEED = 42
MAX_NEW = 70

PROMPTS = [
    "It was a dark and stormy",
    "There was once a little",
    "The old man sat down and",
    "The king had three",
]

# (dominant head, silent head = lowest recovery)
PAIRS = {
    "carroll": ("Carroll", 11, 6),
    "poe":     ("Poe",     14, 1),   # Poe's lowest recovery was H1 (-0.21)
    "grimm":   ("Grimm",   11, 6),
}


def gen(model, tokenizer, prompt):
    return steer_generate(model, tokenizer, prompt,
                          seed=SEED, max_new_tokens=MAX_NEW)


def main():
    tokenizer = load_tokenizer()
    buf = [f"Experiment A probe: keep only one head's LoRA  seed={SEED}  max_new={MAX_NEW}"]

    for key, (label, dom, quiet) in PAIRS.items():
        print(f"[loading {label}]", flush=True)
        template = load_adapted_model(str(ADAPTERS_DIR / key / "adapter"))
        deltas = load_adapter_deltas(str(ADAPTERS_DIR / key / "adapter"))

        buf.append("\n" + "═" * 70)
        buf.append(f"{label}   dominant=H{dom}   silent=H{quiet}")
        buf.append("═" * 70)

        for prompt in PROMPTS:
            print(f"  • {label} | {prompt!r}", flush=True)

            # baseline (full adapter)
            base_text = gen(template, tokenizer, prompt)

            # keep only dominant
            m_dom = copy.deepcopy(template)
            inject_knockout(m_dom, deltas, dom)
            dom_text = gen(m_dom, tokenizer, prompt)
            del m_dom

            # keep only silent
            m_quiet = copy.deepcopy(template)
            inject_knockout(m_quiet, deltas, quiet)
            quiet_text = gen(m_quiet, tokenizer, prompt)
            del m_quiet

            buf.append(f"\nPrompt: {prompt!r}")
            buf.append(f"  full adapter:        {base_text}")
            buf.append(f"  keep only H{dom} (DOM):  {dom_text}")
            buf.append(f"  keep only H{quiet} (quiet): {quiet_text}")

        del template, deltas

    out = "\n".join(buf)
    print(out)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(out)
    print(f"\n[saved to {OUT_PATH}]")


if __name__ == "__main__":
    main()