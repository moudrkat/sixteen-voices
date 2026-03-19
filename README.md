# Sixteen Voices

**A mechanistic interpretability experiment on a tiny transformer**

77 LoRA adapters, 16 heads, 1,232 knockout experiments — all on CPU.

![Per-head knockout recovery across 77 authors](figures/knockout_strip_clean.png)

*Each dot is one author. H11 (blue) leads, H3 (green) is a consistent second, H14 (red) has the widest spread — essential for some authors, actively harmful for others.*

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

**Overview (LinkedIn):** [docs/ARTICLE_SIMPLE.md](docs/ARTICLE_SIMPLE.md)
| **Detailed article:** [docs/ARTICLE_SHORT.md](docs/ARTICLE_SHORT.md)
| **Technical report:** [docs/TECHNICAL.md](docs/TECHNICAL.md)

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

See the article for figures, caveats, and the full argument.

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
— we use heads as the unit of analysis deliberately: with only 16 of
them, you can enumerate everything.

## Future directions

- **Sparse autoencoder on the residual stream** — decompose head
  outputs into interpretable features (`scripts/train_sae.py`, WIP)
- **Hypernetwork** — predict LoRA weights from text, using PCA-compressed
  adapter space (`scripts/train_hypernetwork.py`, WIP)
- **2-layer model** — does the V-Q mechanism survive cross-layer
  composition?

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
```

## Interactive demos

```bash
pip install -e ".[demo]"
streamlit run demos/app_steer.py        # Head knockout lab
streamlit run demos/app_explainer.py    # LoRA + attention explainer
streamlit run demos/app_transplant.py   # Head transplant lab
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
  ARTICLE_SIMPLE.md       # Overview article (LinkedIn)
  ARTICLE_SHORT.md        # Detailed article
  TECHNICAL.md            # Full experiment descriptions
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
