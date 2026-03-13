# TODO — Sixteen Voices

## Done
- [x] Training all 82 adapters
- [x] `fig_head_importance.py` — strip plot + heatmap + correlation from LoRA weight norms
- [x] H14 most variable in v_proj, H6 in q_proj (replicates hyperstories finding)
- [x] Article section 4.1 filled in with weight-norm results
- [x] Article section 2.1 updated — Q/K/V roles explained, why we skip K
- [x] Article limitations updated — pretraining confound, n=1 model
- [x] Article section 4.6 — null baseline subsection added
- [x] LinkedIn post drafted (LINKEDIN.md)
- [x] Run null baseline: `outputs/knockout_null_baseline.json`
- [x] Run steering: `outputs/multihead_text.json` (carroll, poe, grimm, minimalist)
- [x] Run transplant: Poe H14 → Carroll, Grimm, Minimalist (cherry-picked results)
- [x] Transplant figure: `fig_transplant.py` → `figures/transplant.png`
- [x] Article rewritten for LinkedIn (shorter, honest, 7 figures)
- [x] Eval adapters: 82/82 learned (fixed: now uses `extract_prose()` consistently)
- [x] Run knockout: `outputs/knockout_all_heads.json` — H11 dominant (41/82), H14 most polarized
- [x] **Knockout heatmap** — `fig_knockout_heatmap.py` → heatmap, strip plot, best-head bar chart
- [x] Fixed `eval_adapters.py` — was using `clean_text()` instead of `extract_prose()`
- [x] Fixed `train_lora.py` — was using raw text instead of `clean_text()`
- [x] Baker/blake learned fine despite small data (ratio 0.174, 0.075)

## Still to run
- [ ] Run Q/V decomposition: `python scripts/qv_decomposition.py`
- [ ] Run vocab knockout: `python scripts/vocab_knockout.py`
- [ ] Re-run `fig_head_importance.py` with all 82 adapters

## Figures needed
- [ ] Head importance clusters (if heatmap shows structure — don't force it)
- [ ] V-rotation analysis (Q vs V contribution)
- [ ] Do weight norms predict knockout recovery? (scatter: norm vs recovery per head per author)

## Article (ARTICLE.md)
- [ ] Fill in Results sections 4.2–4.5, 4.7 with actual numbers from knockout
- [ ] Check if synthetic controls validate (minimalist vs unusual_vocab)
- [ ] Add knockout heatmap figure
- [ ] Review: is the framing still honest after seeing real results?

## Apps
- [x] `demos/app_steer.py` — head knockout challenge game
- [x] `demos/app_explainer.py` — interactive LoRA & multihead explainer (5 pages)
- [ ] Test explainer app: `streamlit run demos/app_explainer.py`

## Small fixes
- [ ] `app_steer.py` has hardcoded HEAD_NAMES (H3, H11, H13, H14) from old results — update after new knockout
- [ ] `train_all.py` has fragile `from train_lora import train` import (relative to scripts/)
- [ ] Authors with very little data (baker 3k, blake 5k) may have noisy results — consider flagging/excluding

## Maybe later
- [ ] ML Prague poster submission
- [ ] 2-layer comparison experiment (would strengthen the "is this just a 1-layer artifact?" question)
- [ ] PCA of combined Q+V fingerprints (32-dim) — see if authors cluster in weight space