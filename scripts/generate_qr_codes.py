#!/usr/bin/env python3
"""Generate QR codes for the presentation.

Usage:
    uv run python scripts/generate_qr_codes.py

Doplň URL v dictu URLS níže, spusť, a QR kódy skončí v presentation_assets/qr/.

Instalace závislosti:
    uv add 'qrcode[pil]'
    # nebo: pip install 'qrcode[pil]'
"""

from pathlib import Path

import qrcode
from qrcode.constants import ERROR_CORRECT_M


# ═══════════════════════════════════════════════════════════════
# DOPLŇ URL ZDE
# ═══════════════════════════════════════════════════════════════

URLS = {
    # filename bez přípony → URL
    "streamlit_app": "https://sixteen-voices.streamlit.app",
    "linkedin":       "https://linkedin.com/in/katerina-fajmanova",
    "hackathon_factory": "https://hackathon-factory.cz",   # ← oprav až budeš znát skutečnou URL
    "methodology":    "https://github.com/moudrkat/sixteen-voices/blob/main/docs/METHODOLOGY_POSTER.md",
}

# ═══════════════════════════════════════════════════════════════


OUT_DIR = Path("presentation_assets/qr")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_qr(url: str, out_path: Path, box_size: int = 12, border: int = 2):
    """Create a clean black-on-white QR code PNG."""
    qr = qrcode.QRCode(
        version=None,               # auto-size
        error_correction=ERROR_CORRECT_M,  # ~15% error tolerance
        box_size=box_size,
        border=border,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(out_path)
    print(f"  {out_path.name:40s}  {url}")


def main():
    print(f"Generating {len(URLS)} QR codes → {OUT_DIR}/")
    for name, url in URLS.items():
        out = OUT_DIR / f"qr_{name}.png"
        make_qr(url, out)
    print("Done.")


if __name__ == "__main__":
    main()
