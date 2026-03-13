# Sixteen Voices

**Attention head specialization in LoRA-adapted language models**

We adapt a 1-layer, 16-head transformer to 82 authors via LoRA and ask:
*do all heads contribute equally, or does each author route style through
a different subset?* Trained entirely on CPU, because why not.

Some heads matter a lot and others don't — and the pattern is different
per author. It's a toy experiment on one tiny model, but small enough to
see everything.

## The Idea

A tiny transformer has only 16 attention heads. When you teach it 82
different writing styles (via LoRA adapters), each adapter puts different
amounts of weight into different heads. By knocking out one head at a
time and measuring the perplexity change, you get a **16-dimensional
fingerprint** of each adapter — showing which heads carry each author's
adaptation.

## Quick Start

```bash
pip install -e ".[all]"

# Download 69 Gutenberg authors + 13 synthetic styles
make data

# Train all 82 LoRA adapters
make train
```

## Reproduce Key Experiments

```bash
# Evaluate adapters: base vs adapted perplexity
python scripts/eval_adapters.py

# Head importance: 82 authors × 16 heads knockout
python scripts/knockout.py

# Multi-head steering: scale heads at inference
python scripts/steer.py

# Cross-author head transplant
python scripts/transplant.py carroll shelley

# Per-head vocabulary attribution
python scripts/vocab_knockout.py

# Q vs V projection decomposition
python scripts/qv_decomposition.py

# Null baseline (random LoRAs)
python scripts/knockout_null.py
```

## Data

| Source | Authors | Description |
|--------|---------|-------------|
| Project Gutenberg | 69 | Classic literature (Poe, Carroll, Twain, ...) |
| Synthetic | 13 | Controlled styles (minimalist, poet, dialogue, ...) |
| **Total** | **82** | cleaned, max 50k words each |

Text is cleaned (Gutenberg boilerplate, illustration tags, TOC blocks,
frontmatter removed) and truncated to 50k words per author for balanced
training.

Synthetic authors serve as **controls** — their style dimensions are
known by construction, so knockout results can be validated against
ground truth.

## Model

**TinyStories-1Layer-21M** (GPT-Neo architecture)

| Property | Value |
|----------|-------|
| Layers | 1 |
| Hidden dim | 1024 |
| Attention heads | 16 (64 dim each) |
| Vocabulary | 50,257 |
| LoRA rank | 8 (on q_proj + v_proj) |
| Trainable params | 32,768 per adapter (0.15%) |

## Project Structure

```
src/sixteen_voices/     # Importable package
  constants.py          # Model constants
  model.py              # Model loading + LoRA creation
  adapter.py            # LoRA weight manipulation, knockout, SVD
  steering.py           # Attention head steering via hooks
  text.py               # Prose extraction, perplexity, text cleaning
  dataset.py            # Text chunking for training

scripts/                # Experiments & figures
  train_all.py          # Batch training (82 adapters)
  train_lora.py         # Single-author training
  eval_adapters.py      # Base vs adapted perplexity check
  knockout.py           # Core head importance experiment
  knockout_null.py      # Null hypothesis (random LoRAs)
  steer.py              # Multi-head steering
  transplant.py         # Cross-author head transplant
  vocab_knockout.py     # Per-head vocabulary attribution
  qv_decomposition.py   # Q vs V projection analysis
  generate_samples.py   # Reproducible text samples (all authors)
  fig_architecture.py   # Model architecture diagram
  fig_knockout_heatmap.py # Knockout heatmap + strip plot
  fig_head_importance.py  # LoRA weight norm analysis
  fig_transplant.py     # Head transplant comparison figure

data/                   # Download & preprocessing
  get_books.py          # 122 books from Gutenberg
  get_author_datasets.py # Combine into author corpora

figures/                # Generated diagrams
tests/                  # Unit tests (no model download needed)
```

## Figures

Article figures (run `python scripts/fig_*.py`):

- **architecture** — TinyStories-1Layer with 16 heads + LoRA on Q, V
- **knockout_heatmap** — 82 × 16 recovery matrix, clustered
- **knockout_strip** — per-head recovery distribution
- **transplant** — before/after text: Poe's H14 grafted into 3 hosts

## Requirements

- Python 3.10+
- PyTorch, Transformers, PEFT, safetensors
- CPU only — all experiments run in minutes to hours

```bash
pip install -e ".[all]"   # includes matplotlib, pytest, ruff
```

## Article

See [docs/ARTICLE.md](docs/ARTICLE.md) for the full write-up and
[docs/TECHNICAL.md](docs/TECHNICAL.md) for detailed experiment
descriptions with code references.

## Citation

```bibtex
@misc{sixteenvoices2026,
  title   = {Sixteen Voices},
  year    = {2026},
  url     = {https://github.com/moudrkat/sixteen-voices},
}
```

## License

MIT