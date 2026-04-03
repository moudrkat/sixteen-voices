# Sixteen Voices

**A mechanistic interpretability experiment on a tiny transformer**

77 LoRA adapters, 16 heads, 1,232 knockout experiments, a sparse autoencoder — all on CPU.

![Per-head knockout recovery across 77 authors](figures/knockout_strip_clean.png)

*Each dot is one author. H11 (blue) leads, H3 (green) is a consistent second, H14 (red) has the widest spread — essential for some authors, actively harmful for others.*

![How a 1-Layer Transformer Computes Style](figures/sae_heads_roles.png)

---

## What is this?

An interpretability case study. We take the smallest transformer that
still produces coherent text
([TinyStories-1Layer-21M](https://huggingface.co/roneneldan/TinyStories-1Layer-21M),
[Eldan & Li 2023](https://arxiv.org/abs/2305.07759)) and adapt it to
77 different writing styles via [LoRA](https://arxiv.org/abs/2106.09685).
Then we open it up.

With one layer and 16 attention heads, there's nowhere to hide. Every
head gets one shot at the input. You can enumerate all 77 x 16 = 1,232
head-author combinations and ask: **which heads carry which styles, and
why?**

The theoretical backbone is
[Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html) —
in a 1-layer model, each head's contribution to the output is approximately
independent. Our model has an MLP after attention, so the decomposition
is approximate, not exact — but clean enough to trace the chain from
weights to behavior.

**Article 1 — Head knockouts:** [docs/ARTICLE_SIMPLE.md](docs/ARTICLE_SIMPLE.md)
| **Article 2 — SAE features:** [docs/ARTICLE_SAE.md](docs/ARTICLE_SAE.md)
| **Technical reports:** [docs/TECHNICAL.md](docs/TECHNICAL.md), [docs/TECHNICAL_REPORT_SAE.md](docs/TECHNICAL_REPORT_SAE.md)
| **Methodology:** [docs/METHODOLOGY_SAE.md](docs/METHODOLOGY_SAE.md)

## Key findings

1. **Three heads matter, the rest don't.** H11 leads for 66% of authors,
   H14 for 23%, H3 is a consistent second. H11 and H14 are anticorrelated —
   they do the same job (main style carrier) for different author groups.
   This is learned, not random — untrained adapters don't show it.
2. **LoRA changes what heads output, not where they look.** Attention
   patterns are stable across all 77 adapters — style flows through
   value projections, not query/key routing.
3. **V works in isolation, Q doesn't.** V-only beats Q-only for 68/77
   authors (88%). V changes are self-contained; Q changes depend on V.

See Article 1 for figures, caveats, and the full argument.

4. **Style has two layers** (Article 2). A sparse autoencoder (TopK, 2048 features)
   decomposes the residual stream into structural features (sentence length,
   punctuation, line breaks — steerable on any model) and semantic features
   (dialect, atmosphere, character voices — detectable everywhere, steerable
   only with the matching LoRA adapter).
5. **The strongest style direction is invisible to heads.** It emerges from
   MLP multi-head mixing. Weight steering can't reach it. Activation steering can.
6. **LoRAs amplify, they don't create.** 98.8% of features in any adapted
   model already exist in the base model. Style is latent.

## Related work

This builds on head pruning
([Michel et al. 2019](https://arxiv.org/abs/1905.10650),
[Voita et al. 2019](https://arxiv.org/abs/1905.09418)),
activation interventions
([Turner et al. 2023](https://arxiv.org/abs/2308.10248),
[Li et al. 2023](https://arxiv.org/abs/2306.03341)),
and circuit analysis of fine-tuning
([Zhang et al. 2025](https://arxiv.org/abs/2502.11812)).
The field has largely moved to feature-level analysis via sparse
autoencoders
([Bricken et al. 2023](https://transformer-circuits.pub/2023/monosemantic-features),
[Templeton et al. 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/))
— we start with heads (with only 16, you can enumerate everything)
then move to SAE features for finer-grained decomposition.

## Future directions

- **Hypernetwork** — predict LoRA weights from text, using PCA-compressed
  adapter space (`scripts/train_hypernetwork.py`, WIP)
- **2-layer model** — does the V-Q mechanism survive cross-layer
  composition? Does the structural-semantic split hold?
- **Bigger models** — on this 21M model, semantic features only steer
  with the matching adapter. On a bigger model they might steer
  universally. Testable prediction.

## Quick start

```bash
pip install -e ".[all]"

# Download 69 Gutenberg authors + 8 synthetic styles
make data

# Train all 77 LoRA adapters
make train
```

## Reproduce key experiments

```bash
python scripts/eval_adapters.py          # Verify all 77 adapters learned
python scripts/knockout.py               # Core: 77 x 16 head knockout
python scripts/knockout_null.py          # Null baseline (random LoRAs)
python scripts/steering_sweep.py         # Scale heads 0x-2x, measure PPL
python scripts/vq_knockout.py            # V-only vs Q-only isolation
python scripts/transplant.py             # Cross-author head transplant
python scripts/head_attention_patterns.py # Attention pattern analysis
python scripts/attention_stability.py    # Attention stability across adapters
python scripts/retrain_stability.py      # Retraining stability (5 seeds)
python scripts/fig_knockout_heatmap.py   # Generate figures
python scripts/fig_steering.py
# see scripts/fig_*.py for all figure scripts

# Article 2: SAE features
python scripts/train_sae.py --activation topk --k 16 --n-features 2048 --epochs 10 --output outputs/sae_topk16_2048
python scripts/analyze_sae.py --sae-dir outputs/sae_topk16_2048
python scripts/analyze_sae_features_v2.py --sae-dir outputs/sae_topk16_2048
python scripts/sweep_sae_steering_topk.py
python scripts/steer_sae_features.py --sae-dir outputs/sae_topk16_2048 --author poe --features 665:+15
python scripts/steer_sae_features.py --sae-dir outputs/sae_topk16_2048 --features 9:+10 1777:+10 665:+10 --seeds 42 123 456
```

## Interactive demos

```bash
pip install -e ".[demo]"
streamlit run demos/app_features.py    # SAE feature steering lab
streamlit run demos/app_steer.py       # Head knockout lab
streamlit run demos/app_explainer.py   # LoRA + attention explainer
streamlit run demos/app_transplant.py  # Head transplant lab
```

## Model

**TinyStories-1Layer-21M** (GPT-Neo) — 1 layer, 16 heads, 1024 hidden
dim. Each LoRA adapter adds ~33k trainable parameters (0.15% of the
model) on the Q and V projections at rank 8.

## Project structure

```
src/sixteen_voices/       # Library: model loading, steering, knockout
scripts/                  # Experiments + figure generation
demos/                    # Streamlit interactive apps
docs/
  ARTICLE_SIMPLE.md       # Article 1: head knockouts
  ARTICLE_SAE.md          # Article 2: SAE features + steering
  TECHNICAL.md            # Technical report (heads)
  TECHNICAL_REPORT_SAE.md # Technical report (SAE)
  METHODOLOGY_SAE.md      # Pipeline methodology
figures/                  # Generated plots
tests/                    # Unit tests (no model download needed)
```

## Requirements

Python 3.10+, PyTorch, Transformers, PEFT. CPU only — all experiments
run in minutes to hours.

```bash
pip install -e ".[all]"
```

## License

MIT
