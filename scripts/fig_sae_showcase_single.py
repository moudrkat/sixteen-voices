#!/usr/bin/env python3
"""Generate single-author showcase figures for article and LinkedIn.

Produces clean horizontal strips: baseline → feature1 → feature2 → ...

Usage:
    uv run python scripts/fig_sae_showcase_single.py
"""

import json
import re
from pathlib import Path
from textwrap import fill

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_agg import FigureCanvasAgg

from sixteen_voices import load_tokenizer
from sixteen_voices.model import load_adapted_model
from sixteen_voices.sae import SparseAutoencoder

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
SAE_DIR = Path("outputs/sae_topk16_2048")

DARK = "#333333"
GRAY = "#888888"
LIGHT = "#f7f7f7"
RED = "#c44e52"
BLUE = "#4c72b0"
GREEN = "#55a868"
ORANGE = "#e8a735"

SEED = 123


def generate(model, tokenizer, prompt, seed=SEED, max_new=100,
             hook_fn=None):
    if hook_fn:
        handle = model.transformer.ln_f.register_forward_hook(hook_fn)
    torch.manual_seed(seed)
    ids = tokenizer.encode(prompt, return_tensors="pt")
    plen = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new, temperature=0.7,
            do_sample=True, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()
    if hook_fn:
        handle.remove()
    return text


def make_hook(sae, feature_indices, scale):
    w = sae.decoder.weight.detach()
    vec = torch.zeros(w.shape[0])
    for fidx, weight in feature_indices.items():
        vec += weight * scale * w[:, fidx]

    def hook_fn(mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] + vec.to(out[0].device),) + out[1:]
        return out + vec.to(out.device)
    return hook_fn


def truncate(text, max_chars=250):
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "..."


def _measure_text(fig, renderer, text, fontsize, bold=False):
    """Measure text width in figure coordinates."""
    t = fig.text(0, 0, text, fontsize=fontsize, fontfamily="serif",
                 fontstyle="italic", fontweight="bold" if bold else "normal")
    bb = t.get_window_extent(renderer)
    w = bb.width / (fig.get_size_inches()[0] * fig.dpi)
    h = bb.height / (fig.get_size_inches()[1] * fig.dpi)
    t.remove()
    return w, h


def render_highlighted_text(fig, renderer, x_center, y_center, text,
                            pattern, base_color, hl_color, fontsize,
                            line_spacing=1.5):
    """Render wrapped text with regex-matched spans highlighted in hl_color."""
    lines = text.split("\n")

    # Measure line height
    _, char_h = _measure_text(fig, renderer, "Xg", fontsize)
    line_h = char_h * line_spacing

    total_h = line_h * len(lines)
    y_start = y_center + total_h / 2 - char_h / 2

    for li, line in enumerate(lines):
        y = y_start - li * line_h

        # Split line into (text, is_highlighted) segments
        if pattern and line.strip():
            segments = []
            last = 0
            for m in re.finditer(pattern, line):
                if m.start() > last:
                    segments.append((line[last:m.start()], False))
                segments.append((m.group(), True))
                last = m.end()
            if last < len(line):
                segments.append((line[last:], False))
            if not segments:
                segments = [(line, False)]
        else:
            segments = [(line, False)]

        # Measure total line width for centering
        seg_widths = []
        for seg_text, is_hl in segments:
            w, _ = _measure_text(fig, renderer, seg_text, fontsize, bold=is_hl)
            seg_widths.append(w)
        total_w = sum(seg_widths)

        # Render each segment
        x = x_center - total_w / 2
        for (seg_text, is_hl), w in zip(segments, seg_widths):
            color = hl_color if is_hl else base_color
            weight = "bold" if is_hl else "normal"
            fig.text(x, y, seg_text, fontsize=fontsize, fontfamily="serif",
                     fontstyle="italic", color=color, fontweight=weight,
                     ha="left", va="center", transform=fig.transFigure)
            x += w


def render_strip(title, subtitle, rows, output_path):
    """Render a vertical stack figure — one row per feature.

    rows: list of {label, color, text, highlight (regex pattern)}
    """
    n = len(rows)
    fig_w = 14
    header_h = 1.5
    # Compute per-row heights — verse rows get more space
    row_heights = []
    for row in rows:
        if row.get("preserve_breaks"):
            max_lines = row.get("max_lines")
            n_breaks = row["text"].count("\n")
            if max_lines:
                n_breaks = min(n_breaks, max_lines + 1)  # +1 for trailing …
            row_heights.append(max(2.5, 0.35 * n_breaks + 1.5))
        else:
            row_heights.append(2.0)
    fig_h = sum(row_heights) + header_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    canvas = FigureCanvasAgg(fig)
    renderer = canvas.get_renderer()

    # Title and subtitle at the top
    fig.text(0.5, 1 - 0.3 / fig_h, title,
             ha="center", va="top", fontsize=20, fontweight="bold")
    fig.text(0.5, 1 - 0.8 / fig_h, subtitle,
             ha="center", va="top", fontsize=12, color=GRAY)

    y_offset = header_h
    for i, col in enumerate(rows):
        color = col["color"]
        max_chars = 300 if col.get("preserve_breaks") else 200
        text = truncate(col["text"], max_chars)
        is_baseline = (i == 0)

        # Row position (top to bottom, variable height)
        rh = row_heights[i]
        y_top = 1 - y_offset / fig_h
        y_bot = 1 - (y_offset + rh) / fig_h
        y_mid = (y_top + y_bot) / 2
        h = y_top - y_bot
        y_offset += rh

        # Background box (right side, where text goes)
        box = FancyBboxPatch(
            (0.18, y_bot + 0.005), 0.80, h - 0.01,
            boxstyle="round,pad=0.01",
            facecolor=LIGHT if is_baseline else color,
            alpha=0.3 if is_baseline else 0.06,
            edgecolor=GRAY if is_baseline else color,
            linewidth=1.5 if is_baseline else 2.0,
            transform=fig.transFigure,
        )
        fig.patches.append(box)

        # Label on the left
        prefix = "" if is_baseline else "+ "
        fig.text(0.02, y_mid + 0.01, f"{prefix}{col['label']}",
                 ha="left", va="center", fontsize=15,
                 fontweight="bold", color=color)

        # Feature description below label
        if col.get("desc"):
            fig.text(0.02, y_mid - 0.03, col["desc"],
                     ha="left", va="top", fontsize=9.5,
                     color=GRAY, fontstyle="italic")

        # Text — preserve existing line breaks for verse
        if col.get("preserve_breaks") and "\n" in text:
            lines = []
            for paragraph in text.split("\n"):
                paragraph = paragraph.strip()
                if paragraph:
                    lines.append(paragraph)
                else:
                    lines.append("")
            max_lines = col.get("max_lines")
            if max_lines and len(lines) > max_lines:
                lines = lines[:max_lines] + ["", "\u2026"]
            wrapped = "\n".join(lines)
        else:
            wrapped = fill(text, width=75)

        hl_pattern = col.get("highlight")
        if hl_pattern and not is_baseline:
            render_highlighted_text(fig, renderer, 0.58, y_mid, wrapped,
                                    hl_pattern, DARK, color, 12.5)
        else:
            fig.text(0.58, y_mid, wrapped, ha="center", va="center",
                     fontsize=12.5, color=DARK, fontfamily="serif",
                     fontstyle="italic", linespacing=1.5)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    with open(SAE_DIR / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(SAE_DIR / "sae_weights.pt", config)
    tokenizer = load_tokenizer()

    # ── Carroll: 4 features ──
    # Each feature uses the seed that produces the clearest result
    print("Carroll...")
    model = load_adapted_model("outputs/authors/carroll/adapter")
    prompt = "Alice was beginning to get very"

    bl = generate(model, tokenizer, prompt)
    s1 = generate(model, tokenizer, prompt,
                  hook_fn=make_hook(sae, {665: 1.0}, 15.0))
    s2 = generate(model, tokenizer, prompt, seed=42,
                  hook_fn=make_hook(sae, {1779: 1.0}, 15.0))
    s3 = generate(model, tokenizer, prompt,
                  hook_fn=make_hook(sae, {1777: 1.0, 689: 1.0}, 8.0))
    s4 = generate(model, tokenizer, prompt, seed=456,
                  hook_fn=make_hook(sae, {344: 1.0}, 20.0),
                  max_new=120)

    render_strip(
        "Turning the Knobs Inside a Tiny Language Model",
        "Model fine-tuned on Lewis Carroll (Alice in Wonderland). Same prompt — internal features control the output style.",
        [
            {"label": "baseline", "color": GRAY, "text": bl},
            {"label": "simplicity", "color": RED, "text": s1,
             "desc": "sentence-ending periods",
             "highlight": r"\."},
            {"label": 'first-person "I"', "color": BLUE, "text": s2,
             "desc": "the pronoun I",
             "highlight": r"\bI\b"},
            {"label": "dialogue", "color": GREEN, "text": s3,
             "desc": '"said the..." patterns',
             "highlight": r'"[^"]*"|said|asked|cried'},
            {"label": "verse", "color": ORANGE, "text": s4,
             "desc": "verse line breaks",
             "preserve_breaks": True, "max_lines": 3},
        ],
        FIGURES_DIR / "sae_showcase_carroll.png",
    )

    # ── Poe: 2 features ──
    print("Poe...")
    model = load_adapted_model("outputs/authors/poe/adapter")
    prompt = "There came a midnight"

    bl = generate(model, tokenizer, prompt, seed=42)
    s1 = generate(model, tokenizer, prompt, seed=42,
                  hook_fn=make_hook(sae, {665: 1.0}, 15.0))
    s2 = generate(model, tokenizer, prompt, seed=42,
                  hook_fn=make_hook(sae, {1779: 1.0}, 12.0))

    render_strip(
        "Poe — Two Directions",
        "Model fine-tuned on Edgar Allan Poe. Same prompt — internal features control the output style.",
        [
            {"label": "baseline", "color": GRAY, "text": bl},
            {"label": "simplicity", "color": RED, "text": s1,
             "desc": "sentence-ending periods",
             "highlight": r"\."},
            {"label": 'first-person "I"', "color": BLUE, "text": s2,
             "desc": "the pronoun I",
             "highlight": r"\bI\b"},
        ],
        FIGURES_DIR / "sae_showcase_poe.png",
    )

    print("Done!")


if __name__ == "__main__":
    main()