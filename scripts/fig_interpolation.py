#!/usr/bin/env python3
"""Generate interpolation figure: blend two authors' LoRA adapters.

Produces a LinkedIn-style figure showing text samples at different
interpolation points with a perplexity curve, plus saves raw data.

Usage:
    python scripts/fig_interpolation.py
"""

import copy
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch

from sixteen_voices.model import (
    load_adapted_model,
    load_tokenizer,
    get_attn_module,
)
from sixteen_voices.adapter import load_adapter_deltas, delta_to_AB

ADAPTERS_DIR = Path("outputs/authors")
OUT_DIR = Path("outputs")
FIG_DIR = Path("figures")

# --- Config ---
AUTHOR_A = "carroll"
AUTHOR_B = "poet"
PROMPT = "It was a dark and stormy"
SEED = 42
MAX_NEW = 80
N_STEPS = 11  # 0.0, 0.1, ..., 1.0

EVAL_TEXT = (
    "Once upon a time there was a little girl who lived in a small house "
    "near the forest with her mother and father and a cat named Whiskers"
)

# Alphas to show as text boxes
SHOW_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

FONT_PROSE = "Noto Serif Display"
C_POE = "#C44E52"
C_CARROLL = "#2980B9"
C_TEXT = "#333333"


def inject_deltas(model, deltas):
    attn = get_attn_module(model)
    for proj in ["q_proj", "v_proj"]:
        A, B = delta_to_AB(deltas[proj])
        getattr(attn, proj).lora_A["default"].weight.data.copy_(A)
        getattr(attn, proj).lora_B["default"].weight.data.copy_(B)


def interpolate_deltas(d1, d2, alpha):
    return {
        proj: (1 - alpha) * d1[proj] + alpha * d2[proj]
        for proj in ["q_proj", "v_proj"]
    }


def compute_ppl(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(out.loss).item()


def generate(model, tokenizer, prompt, seed=42, max_new=80):
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt")
    plen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()


def run_experiment():
    """Run interpolation, return results list."""
    print("Loading tokenizer and models...")
    tokenizer = load_tokenizer()
    template_model = load_adapted_model(
        str(ADAPTERS_DIR / AUTHOR_A / "adapter"))

    d_a = load_adapter_deltas(str(ADAPTERS_DIR / AUTHOR_A / "adapter"))
    d_b = load_adapter_deltas(str(ADAPTERS_DIR / AUTHOR_B / "adapter"))

    alphas = [round(i / (N_STEPS - 1), 2) for i in range(N_STEPS)]

    results = []
    for alpha in alphas:
        print(f"  alpha = {alpha:.2f}...")
        blended = interpolate_deltas(d_a, d_b, alpha)
        model = copy.deepcopy(template_model)
        inject_deltas(model, blended)

        text = generate(model, tokenizer, PROMPT, seed=SEED, max_new=MAX_NEW)
        ppl = compute_ppl(model, tokenizer, EVAL_TEXT)
        results.append({"alpha": alpha, "text": text, "ppl": ppl})
        del model

    # Save JSON
    out_data = {
        "author_a": AUTHOR_A,
        "author_b": AUTHOR_B,
        "prompt": PROMPT,
        "seed": SEED,
        "eval_text": EVAL_TEXT,
        "results": results,
    }
    out_path = OUT_DIR / "interpolation_samples.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved {out_path}")
    return results


def blend_color(alpha):
    """Blend from Carroll blue to Poet purple."""
    r_a = (0.16, 0.50, 0.72)
    r_b = (0.58, 0.30, 0.65)
    return tuple((1 - alpha) * a + alpha * b for a, b in zip(r_a, r_b))


def make_figure(results):
    """Create LinkedIn-optimized interpolation figure."""
    alphas_all = [r["alpha"] for r in results]
    ppls_all = [r["ppl"] for r in results]

    # Pick closest actual results to show
    show = []
    for target in SHOW_ALPHAS:
        closest = min(results, key=lambda r: abs(r["alpha"] - target))
        show.append(closest)

    n_show = len(show)
    # Last box is taller to show full text
    box_ratios = [2.2] * (n_show - 1) + [5.5]
    fig_h = 3.2 + (n_show - 1) * 2.6 + 6.5
    fig, axes = plt.subplots(n_show + 1, 1, figsize=(14, fig_h),
                             gridspec_kw={"height_ratios": [2.5] + box_ratios,
                                          "hspace": 0.2})

    # ── PPL curve ──
    ax = axes[0]
    ax.plot(alphas_all, ppls_all, "o-", color="#888888", linewidth=2,
            markersize=5, zorder=3)
    for r in show:
        c = blend_color(r["alpha"])
        ax.scatter([r["alpha"]], [r["ppl"]], color=c, s=120, zorder=5,
                   edgecolors="white", linewidth=2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title(
        f"Interpolating LoRA adapters:  {AUTHOR_A.capitalize()} → {AUTHOR_B.capitalize()}",
        fontsize=16, fontweight="bold", pad=10)
    ax.set_xlabel(
        f"← {AUTHOR_A.capitalize()}                    α                    "
        f"{AUTHOR_B.capitalize()} →", fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Text boxes ──
    for i, r in enumerate(show):
        ax = axes[i + 1]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        alpha = r["alpha"]
        text = r["text"]
        ppl = r["ppl"]
        color = blend_color(alpha)
        is_last = (i == n_show - 1)

        # Background box
        bg_color = tuple(list(color) + [0.12])
        ax.add_patch(FancyBboxPatch(
            (0.01, 0.05), 0.98, 0.9, boxstyle="round,pad=0.02",
            facecolor=bg_color, edgecolor=color, linewidth=2,
            transform=ax.transAxes))

        # Label
        label = f"α = {alpha:.2f}"
        if alpha == 0:
            label += f"  (pure {AUTHOR_A.capitalize()})"
        elif alpha == 1:
            label += f"  (pure {AUTHOR_B.capitalize()})"
        ax.text(0.03, 0.85, label, fontsize=12, fontweight="bold",
                color=color, transform=ax.transAxes, va="top")

        # PPL
        ax.text(0.97, 0.85, f"PPL: {ppl:.1f}", fontsize=10,
                color="#999999", transform=ax.transAxes, va="top", ha="right")

        # Text — preserve original line breaks, only wrap long lines
        truncated = text if is_last else text[:250]
        lines = truncated.split("\n")
        wrapped_lines = []
        for line in lines:
            line = line.strip()
            if line == "---":
                wrapped_lines.append("—  —  —")
            elif len(line) > 110:
                wrapped_lines.extend(textwrap.wrap(line, width=110))
            elif line:
                wrapped_lines.append(line)
        max_lines = 12 if is_last else 5
        display_text = "\n".join(wrapped_lines[:max_lines])

        ax.text(0.03, 0.62, display_text, fontsize=10.5, color=C_TEXT,
                style="italic", fontfamily=FONT_PROSE,
                transform=ax.transAxes, va="top", linespacing=1.4)

    # Footer
    fig.text(0.5, 0.005,
             f'Prompt: "{PROMPT}" · seed={SEED} · TinyStories-1Layer-21M · '
             f'LoRA rank 8 · linear interpolation in weight space',
             ha="center", fontsize=9, color="#aaaaaa")

    fig_path = FIG_DIR / "interpolation.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {fig_path}")

    # LinkedIn version — same 5 points, last box bigger for full text
    fig_path = FIG_DIR / "interpolation_linkedin.png"
    fig_h = 3.2 + (n_show - 1) * 2.6 + 6.5
    fig, axes = plt.subplots(n_show + 1, 1, figsize=(14, fig_h),
                             gridspec_kw={"height_ratios": [2.5] + box_ratios,
                                          "hspace": 0.2})
    ax = axes[0]
    ax.plot(alphas_all, ppls_all, "o-", color="#888888", linewidth=2,
            markersize=5, zorder=3)
    for r in show:
        c = blend_color(r["alpha"])
        ax.scatter([r["alpha"]], [r["ppl"]], color=c, s=120, zorder=5,
                   edgecolors="white", linewidth=2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title(
        f"Interpolating LoRA adapters:  {AUTHOR_A.capitalize()} → {AUTHOR_B.capitalize()}",
        fontsize=16, fontweight="bold", pad=10)
    ax.set_xlabel(
        f"← {AUTHOR_A.capitalize()}                    α                    "
        f"{AUTHOR_B.capitalize()} →", fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, r in enumerate(show):
        ax = axes[i + 1]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        alpha_val = r["alpha"]
        text = r["text"]
        ppl = r["ppl"]
        color = blend_color(alpha_val)
        is_last = (i == n_show - 1)
        bg_color = tuple(list(color) + [0.12])
        ax.add_patch(FancyBboxPatch(
            (0.01, 0.05), 0.98, 0.9, boxstyle="round,pad=0.02",
            facecolor=bg_color, edgecolor=color, linewidth=2,
            transform=ax.transAxes))
        label = f"α = {alpha_val:.2f}"
        if alpha_val == 0:
            label += f"  (pure {AUTHOR_A.capitalize()})"
        elif alpha_val == 1:
            label += f"  (pure {AUTHOR_B.capitalize()})"
        ax.text(0.03, 0.85, label, fontsize=12, fontweight="bold",
                color=color, transform=ax.transAxes, va="top")
        ax.text(0.97, 0.85, f"PPL: {ppl:.1f}", fontsize=10,
                color="#999999", transform=ax.transAxes, va="top", ha="right")
        # Preserve line breaks
        truncated = text if is_last else text[:250]
        lines = truncated.split("\n")
        wrapped_lines = []
        for line in lines:
            line = line.strip()
            if line == "---":
                wrapped_lines.append("—  —  —")
            elif len(line) > 110:
                wrapped_lines.extend(textwrap.wrap(line, width=110))
            elif line:
                wrapped_lines.append(line)
        max_lines = 12 if is_last else 5
        display_text = "\n".join(wrapped_lines[:max_lines])
        ax.text(0.03, 0.62, display_text, fontsize=10.5, color=C_TEXT,
                style="italic", fontfamily=FONT_PROSE,
                transform=ax.transAxes, va="top", linespacing=1.4)

    fig.text(0.5, 0.005,
             f'Prompt: "{PROMPT}" · seed={SEED} · TinyStories-1Layer-21M · '
             f'LoRA rank 8 · linear interpolation in weight space',
             ha="center", fontsize=9, color="#aaaaaa")
    fig_path = FIG_DIR / "interpolation_linkedin.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {fig_path}")


def main():
    # Check if we already have data
    json_path = OUT_DIR / "interpolation_samples.json"
    if json_path.exists():
        print(f"Found existing {json_path}, using cached data.")
        with open(json_path) as f:
            data = json.load(f)
        results = data["results"]
    else:
        results = run_experiment()

    make_figure(results)


if __name__ == "__main__":
    main()