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


if __name__ == "__main__":
    make_alice()
    make_poem()
    make_poem_en()
