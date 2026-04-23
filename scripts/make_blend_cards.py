"""Generate the two text-card images for the Blend tab of app_presentation.py.

- alice_card.png : Alice in Wonderland opening, with "Alice" as a colorful title line
- poem_card.png  : free-verse (non-rhyming) poem stacked as lines
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "presentation_assets" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# Presentation palette
CREAM = "#FEF3C7"
ACC_PURPLE = "#7C3AED"
ACC_BLUE = "#2563EB"
ACC_ORANGE = "#EA580C"
FEAT_TEAL = "#0D9488"
ACC_BLACK = "#1A1A1A"
MUTED = "#666666"
PALE_TEAL = "#CCFBF1"


def _card(fig, ax, bg):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)


def make_alice():
    fig, ax = plt.subplots(figsize=(6.4, 4.5), dpi=200)
    _card(fig, ax, CREAM)

    # Decorative border accent (thin corner marks)
    for (x, y) in [(0.35, 6.55), (9.65, 6.55), (0.35, 0.35), (9.65, 0.35)]:
        ax.plot([x - 0.15, x + 0.15], [y, y], color=ACC_ORANGE, lw=1.2)
        ax.plot([x, x], [y - 0.15, y + 0.15], color=ACC_ORANGE, lw=1.2)

    # Title line: "Alice"
    ax.text(
        5, 5.55, "Alice",
        ha="center", va="center",
        fontsize=54, fontweight="bold",
        fontfamily="serif", fontstyle="italic",
        color=ACC_PURPLE,
    )

    # Passage: each line manually broken for rhythm
    lines = [
        "was beginning to get very tired of sitting",
        "by her sister on the bank, and of having",
        "nothing to do: once or twice she had peeped",
        "into the book her sister was reading, but it",
        "had no pictures or conversations in it,",
        "'and what is the use of a book,' thought Alice,",
        "'without pictures or conversations?'",
    ]
    y = 4.05
    for ln in lines:
        ax.text(
            5, y, ln,
            ha="center", va="center",
            fontsize=13, fontfamily="serif",
            color=ACC_BLACK,
        )
        y -= 0.45

    # Attribution
    ax.text(
        5, 0.75, "— Lewis Carroll",
        ha="center", va="center",
        fontsize=11, fontstyle="italic",
        color=MUTED,
    )

    plt.tight_layout(pad=0.2)
    out = OUT / "alice_card.png"
    fig.savefig(out, dpi=200, facecolor=CREAM, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out}")


def make_poem():
    fig, ax = plt.subplots(figsize=(6.4, 4.5), dpi=200)
    _card(fig, ax, PALE_TEAL)

    # Corner accents
    for (x, y) in [(0.35, 6.55), (9.65, 6.55), (0.35, 0.35), (9.65, 0.35)]:
        ax.plot([x - 0.15, x + 0.15], [y, y], color=FEAT_TEAL, lw=1.2)
        ax.plot([x, x], [y - 0.15, y + 0.15], color=FEAT_TEAL, lw=1.2)

    # Title: Básník
    ax.text(
        5, 5.9, "Básník",
        ha="center", va="center",
        fontsize=38, fontweight="bold",
        fontfamily="serif", fontstyle="italic",
        color=ACC_BLUE,
    )

    # Free-verse (non-rhyming), stacked lines with varying indent
    # English to match Alice card being English
    poem = [
        ("a page turns", 0.0),
        ("and the light falls", 0.6),
        ("between the lines", 1.2),
        ("", 0.0),
        ("nothing rhymes", 0.3),
        ("but every word", 0.9),
        ("still breathes", 1.5),
    ]
    y = 4.55
    for line, indent in poem:
        if line:
            ax.text(
                3.2 + indent, y, line,
                ha="left", va="center",
                fontsize=16, fontfamily="serif",
                color=ACC_BLACK,
            )
        y -= 0.5

    plt.tight_layout(pad=0.2)
    out = OUT / "poem_card.png"
    fig.savefig(out, dpi=200, facecolor=PALE_TEAL, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out}")


def make_poem_en():
    """English variant with 'Poet' title for the poster (ML Prague) app."""
    fig, ax = plt.subplots(figsize=(6.4, 4.5), dpi=200)
    _card(fig, ax, PALE_TEAL)

    for (x, y) in [(0.35, 6.55), (9.65, 6.55), (0.35, 0.35), (9.65, 0.35)]:
        ax.plot([x - 0.15, x + 0.15], [y, y], color=FEAT_TEAL, lw=1.2)
        ax.plot([x, x], [y - 0.15, y + 0.15], color=FEAT_TEAL, lw=1.2)

    ax.text(
        5, 5.9, "Poet",
        ha="center", va="center",
        fontsize=44, fontweight="bold",
        fontfamily="serif", fontstyle="italic",
        color=ACC_BLUE,
    )

    poem = [
        ("a page turns", 0.0),
        ("and the light falls", 0.6),
        ("between the lines", 1.2),
        ("", 0.0),
        ("nothing rhymes", 0.3),
        ("but every word", 0.9),
        ("still breathes", 1.5),
    ]
    y = 4.55
    for line, indent in poem:
        if line:
            ax.text(
                3.2 + indent, y, line,
                ha="left", va="center",
                fontsize=16, fontfamily="serif",
                color=ACC_BLACK,
            )
        y -= 0.5

    plt.tight_layout(pad=0.2)
    out = OUT / "poem_card_en.png"
    fig.savefig(out, dpi=200, facecolor=PALE_TEAL, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out}")


def make_blend_diagram():
    """Minimal conceptual diagram: authors as points, blend = line
    between Carroll and Poet with α=0.5 midpoint."""
    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Points (x, y, color, label, label_offset)
    pts = [
        ("Carroll", 1.1, 2.6, ACC_BLUE, (-0.05, 0.15)),
        ("Poet",    3.6, 2.9, "#DC2626", (0.05, 0.15)),
        ("Homer",   1.3, 1.3, "#EA580C", (0.0, -0.35)),
        ("Grimm",   2.5, 1.4, "#16A34A", (0.05, 0.18)),
    ]

    # Dashed blend line Carroll ↔ Poet
    (x1, y1) = (pts[0][1], pts[0][2])
    (x2, y2) = (pts[1][1], pts[1][2])
    ax.plot([x1, x2], [y1, y2],
            linestyle=(0, (5, 4)), color=ACC_PURPLE, lw=1.6, zorder=1)

    # Midpoint hollow circle: α=0.5
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.scatter([mx], [my], s=170, facecolor="white",
               edgecolor=ACC_PURPLE, linewidths=2.0, zorder=3)
    ax.text(mx + 0.05, my - 0.30, r"$\alpha=0.5$",
            color=ACC_PURPLE, fontsize=11, ha="center", va="top")

    # Author points
    for name, x, y, color, (dx, dy) in pts:
        ax.scatter([x], [y], s=90, color=color, zorder=4)
        ax.text(x + dx, y + dy, name, color=color,
                fontsize=12, ha="center", va="center", fontweight="normal")

    # Clean axes with arrows, no ticks
    ax.set_xlim(0.2, 4.2)
    ax.set_ylim(0.5, 3.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888")
    ax.spines["bottom"].set_color("#888")
    ax.tick_params(axis="both", which="both",
                   bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    # Axis arrows
    ax.annotate("", xy=(4.15, 0.55), xytext=(0.25, 0.55),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))
    ax.annotate("", xy=(0.25, 3.35), xytext=(0.25, 0.55),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))

    plt.tight_layout(pad=0.3)
    out = OUT / "blend_diagram.png"
    fig.savefig(out, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_alice()
    make_poem()
    make_poem_en()
    make_blend_diagram()
