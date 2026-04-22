#!/usr/bin/env python3
"""All-16-heads steering curves for a single author.

Reads outputs/steering_sweep_<author>_all.json (from steering_sweep.py
with --all-heads) and renders:

  - figures/<author>_all_heads_curves.png    (all 16 on one axis)
  - figures/<author>_all_heads_grid.png      (4x4 small-multiples grid)

Usage:
    uv run --extra viz python scripts/fig_carroll_all_heads.py
    uv run --extra viz python scripts/fig_carroll_all_heads.py --input outputs/steering_sweep_poe_all.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("figures")

C_H11 = "#2563EB"     # blue — universal backbone
C_H14 = "#DC2626"     # red — formality / elevated
C_H3  = "#16A34A"     # green — universal reader
C_OTH = "#9CA3AF"     # grey — everything else
C_TXT = "#333333"
C_MUT = "#666666"


def head_color(h: int, dominant: int) -> str:
    if h == 11:
        return C_H11
    if h == 14:
        return C_H14
    if h == 3:
        return C_H3
    return C_OTH


def head_style(h: int, dominant: int) -> dict:
    # Always fixed colors for H3/H11/H14, grey for the rest.
    # Dominant head gets thicker line regardless of which it is.
    if h in (3, 11, 14):
        lw = 2.6 if h == dominant else 1.6
        alpha = 1.0 if h == dominant else 0.9
        zorder = 5 if h == dominant else 3
        suffix = " (dom)" if h == dominant else ""
        return dict(color=head_color(h, dominant), lw=lw, alpha=alpha,
                    zorder=zorder, label=f"H{h}{suffix}")
    return dict(color=C_OTH, lw=1.0, alpha=0.55, zorder=1)


def load_all(path: Path):
    with open(path) as f:
        data = json.load(f)
    return list(data.items())


def single_plot(author: str, d: dict):
    rec = d["head_recovery"]
    dom = int(max(rec, key=rec.get)[1:])
    full_ppl = d["full_ppl"]
    curves = d["curves"]

    scales = sorted([float(s) for s in next(iter(curves.values())).keys()])
    scale_labels = [str(s) for s in scales]

    fig, ax = plt.subplots(figsize=(9, 5.2))

    # Grey band of "other heads" (everything except H3/H11/H14)
    for h_key, curve in curves.items():
        h = int(h_key[1:])
        if h in (3, 11, 14):
            continue
        ys = [curve[str(s)] for s in scale_labels]
        ax.plot(scales, ys, **head_style(h, dom))

    # Colored H3/H11/H14 on top
    for h in (3, 11, 14):
        h_key = f"H{h}"
        if h_key not in curves:
            continue
        ys = [curves[h_key][str(s)] for s in scale_labels]
        ax.plot(scales, ys, marker="o", markersize=4, **head_style(h, dom))

    ax.axvline(1.0, color="#dddddd", lw=0.8, zorder=0)
    ax.axhline(full_ppl, color="#dddddd", lw=0.8, ls="--", zorder=0)

    ax.set_xlabel("Head scale", fontsize=11)
    ax.set_ylabel("Perplexity", fontsize=11)
    ax.set_title(
        f"{author.capitalize()} — scaling one attention head at inference",
        fontsize=13, fontweight="bold", color=C_TXT, pad=12)

    leg = ax.legend(loc="upper center", fontsize=9, frameon=False, ncol=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=C_MUT)

    fig.text(0.5, -0.01,
             f"16 heads swept 0× to 2× · PPL on {author} eval text · "
             f"dominant head H{dom} (recovery {rec[f'H{dom}']:.2f}) · "
             f"grey = other 13 heads",
             ha="center", fontsize=8.5, color=C_MUT)

    out = FIG_DIR / f"{author}_all_heads_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def grid_plot(author: str, d: dict):
    rec = d["head_recovery"]
    dom = int(max(rec, key=rec.get)[1:])
    full_ppl = d["full_ppl"]
    curves = d["curves"]

    scales = sorted([float(s) for s in next(iter(curves.values())).keys()])
    scale_labels = [str(s) for s in scales]

    all_ys = [c[s] for c in curves.values() for s in scale_labels]
    ymin, ymax = min(all_ys), max(all_ys)
    pad = (ymax - ymin) * 0.08
    ylim = (ymin - pad, ymax + pad)

    fig, axes = plt.subplots(4, 4, figsize=(11, 9),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.25})

    fig.suptitle(
        f"{author.capitalize()} — steering each of the 16 heads (0× to 2×)",
        fontsize=14, fontweight="bold", color=C_TXT, y=0.995)

    for h in range(16):
        ax = axes[h // 4, h % 4]
        curve = curves[f"H{h}"]
        ys = [curve[str(s)] for s in scale_labels]
        style = head_style(h, dom)
        ax.plot(scales, ys, marker="o", markersize=3,
                color=style["color"], lw=style["lw"], alpha=style["alpha"])

        ax.axvline(1.0, color="#dddddd", lw=0.6)
        ax.axhline(full_ppl, color="#dddddd", lw=0.6, ls="--")

        tag = ""
        if h == dom: tag = " (dom)"
        elif h == 14: tag = " (H14)"
        elif h == 3:  tag = " (H3)"
        ax.set_title(f"H{h}{tag}  rec={rec[f'H{h}']:+.2f}",
                     fontsize=9.5, color=style["color"],
                     fontweight="bold" if h == dom else "normal")
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=7, colors=C_MUT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(0.5, 0.005,
             f"PPL vs head scale · {author} eval text · "
             f"dashed = full adapter ({full_ppl:.1f})",
             ha="center", fontsize=8.5, color=C_MUT)

    out = FIG_DIR / f"{author}_all_heads_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/steering_sweep_carroll_all.json")
    args = parser.parse_args()
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run steering_sweep.py first.")
    for author, d in load_all(path):
        single_plot(author, d)
        grid_plot(author, d)


if __name__ == "__main__":
    main()
