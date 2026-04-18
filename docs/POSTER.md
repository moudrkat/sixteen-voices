# Poster: Sixteen Voices

ML Prague 2026 poster companion. This page indexes everything that sits behind the printed poster.

**Poster PDF:** [`poster.pdf`](../poster.pdf) · **Source:** [`poster.tex`](../poster.tex)

---

## The 30-second pitch

> A tiny 21M-parameter language model. One transformer layer. 16 attention heads. I fine-tuned **77 LoRA adapters** on it, one per author (64 real Project Gutenberg authors + 13 synthetic style archetypes), then asked eight questions about how style lives inside a network that small. All on a laptop CPU.
>
> **The punchline:** style lives in two layers. *Structural* knobs (simplicity, dialogue, verse) steer on any adapter. *Semantic* fingerprints (dark atmosphere, cozy food, dialect) detect everywhere but only steer with the matching adapter. LoRAs amplify, they don't create. And the strongest style direction lives in the MLP, invisible to any single attention head.

---

## Try it yourself

Three small Streamlit demos, all CPU-only, mirror the three marker boxes on the poster:

| Demo | What it does | Link |
|---|---|---|
| **Kill a head** | Pick Carroll or Poe, kill the dominant head, watch the style collapse. (Q1, Q2) | `demos/app_poster_steer.py` · [sixteen-voices-steer.streamlit.app](https://sixteen-voices-steer.streamlit.app) |
| **Blend two authors** | Drag α from 0 to 1 between Carroll and Poet, see the voice morph. (Q4) | `demos/app_poster_blend.py` · [sixteen-voices-blend.streamlit.app](https://sixteen-voices-blend.streamlit.app) |
| **Steer one feature** | Pick an author, drag the simplicity slider, watch gothic prose turn into kindergarten sentences. (Q6) | `demos/app_poster.py` · [sixteen-voices.streamlit.app](https://sixteen-voices.streamlit.app) |

---

## Documentation

Five docs sit behind the poster. Read them in this order if you want the full story:

### 1. [`FAQ.md`](FAQ.md) — common session questions, pre-answered

Start here if you have 5 minutes. Covers:
- How the "leading head" is actually calculated (argmax of recovery, honest caveats)
- What the knockout strip plot shows and doesn't show
- Whether the "other 13 heads" are redundant (short answer: unknown, not tested)
- What the Q2 steering contrast plot shows
- What "dark atmosphere" actually fires on (spoiler: not a pure gothic detector)
- Why the Q4 perplexity curve dips mid-blend (not evidence the blend is "better")

### 2. [`METHODOLOGY_POSTER.md`](METHODOLOGY_POSTER.md) — per-Q methodology with code pointers

Read this if someone asks *"how exactly did you measure X?"*. For each Q1–Q8:
- Procedure (what was done, code file, line pointers)
- Metric definitions (recovery score, perplexity, closed-loop win rate)
- Honest caveats and limits
- Data file locations

Plus shared setup: LoRA training config (rank 8, α 32, q_proj/v_proj only, 8 epochs AdamW at 5e-4), SVD reconstruction math for knockout/transplant, eval-vs-training data split.

### 3. [`AUTHORS.md`](AUTHORS.md) — who's in the 77

Full list of the 64 real authors (with source texts) and 13 synthetic style archetypes (with isolated axis). Also explains the 7 authors dropped for contamination and the training-vs-eval text split.

### 4. [`FEATURE_CATALOG.md`](FEATURE_CATALOG.md) — SAE feature details

For the Q5/Q6 curious. Each of the ~25 interpretable SAE features with firing rate, top-associated authors, steering test results (does it steer? on what models? with what effect?).

### 5. [`METHODOLOGY_SAE.md`](METHODOLOGY_SAE.md) — the SAE pipeline end-to-end

SAE training config, grounded labeling pipeline (synthetic-author cross-check), head-to-feature attribution.

---

## The two articles behind the poster

1. **Article 1 — "Sixteen Voices"** ([LinkedIn](https://www.linkedin.com/pulse/sixteen-voices-interpretability-experiment-tiny-kate%C5%99ina-fajmanov%C3%A1-jmfnf)) · head knockout, steering, transplants, blending. Corresponds to Q1–Q4 on the poster. Source: [`docs/ARTICLE_SIMPLE.md`](ARTICLE_SIMPLE.md).
2. **Article 2 — "Experiment in a Pocket"** ([LinkedIn](https://www.linkedin.com/pulse/experiment-pocket-opening-tiny-model-finding-knobs-kate%C5%99ina-fajmanov%C3%A1-crodf)) · SAE features, feature steering, detection ≠ steering, the MLP direction. Corresponds to Q5–Q8 on the poster. Source: [`docs/ARTICLE_SAE.md`](ARTICLE_SAE.md).

---

## Quick pitch guide for the session

If you're talking someone through the poster from cold:

1. **Hook** (10 sec): *"Tiny language model. 21M params. One layer. 16 heads. 77 LoRA adapters on my laptop. Eight questions."*
2. **Part I** (30 sec): *"Head knockout. Most authors lean on H11. A cluster of elevated writers — Homer, Milton, Poe — leans on H14 instead. H3 is everyone's silent backup."*
3. **Q2 demo** (optional, 20 sec): *"Kill one head, watch the style vanish. Carroll without H11 is a generic rabbit story. Live demo on my phone."*
4. **Part II** (30 sec): *"Then I trained an SAE on the residual stream. Found ~25 features — simplicity, dialogue, first-person, verse. These are the knobs the heads read from."*
5. **Q8 punchline** (15 sec): *"Style lives in two layers. Structural knobs steer anywhere. Semantic ones only amplify with the matching adapter. LoRAs don't create — they amplify. And the strongest direction lives in the MLP, invisible to any head."*

Total: under 2 minutes. Enough time for them to then ask you questions, which is the point.

---

## Repo

All code, data, and experiments: [github.com/moudrkat/sixteen-voices](https://github.com/moudrkat/sixteen-voices)

Top-level scripts of interest:
- `scripts/knockout.py` — head knockout sweep
- `scripts/transplant.py` — head transplant
- `scripts/fig_interpolation.py` — LoRA blend sweep
- `scripts/train_sae.py` — SAE training
- `scripts/sweep_sae_steering.py` — feature steering validation
