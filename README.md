# Sixteen Voices

**Attention head specialization in LoRA-adapted language models**

We adapt a 1-layer, 16-head transformer to 69 authors via LoRA and ask:
*do all heads contribute equally, or does each author route style through
a different subset?*

It turns out **every author gets routed through different heads.**

- **H11** is the universal grammar backbone (most important for 52% of authors)
- **H14** is the great polarizer — complex-vocab authors depend on it, simple-vocab authors suppress it
- The mechanism is **V-rotation**: attention patterns stay the same, but head outputs point to entirely different vocabularies
- Head specialization is **learned** (random LoRAs show no consistent head roles)
- **No single head is "the style head"** — style emerges from the full 16-head routing decision

## Quick Start

```bash
pip install -e .

# Download 69 author texts from Project Gutenberg (~20 min)
make data

# Train all 69 LoRA adapters (~2 hours on CPU)
make train

# Or download pre-trained adapters:
# make adapters
```

## Reproduce Key Results

```bash
# Head importance: 69 authors x 16 heads knockout
python scripts/knockout.py

# Multi-head steering: scale heads at inference
python scripts/steer.py

# Cross-author head transplant
python scripts/transplant.py carroll shelley

# Per-head vocabulary attribution
python scripts/vocab_knockout.py
```

## Key Findings

### 1. H11 is the universal workhorse

Single-head knockout across 69 authors: keep only one head's LoRA contribution
and measure perplexity recovery. H11 alone recovers the most for 36/69 authors
(mean recovery 0.39).

### 2. H14 polarizes on word complexity

H14's importance correlates with vocabulary complexity (r=0.47, p<0.0001).
Complex-vocab authors (Gibbon, Shelley, Lovecraft) route heavily through H14.
Simple-vocab authors (Twain, Potter, Grimm) actively suppress it.

### 3. V-rotation: same eyes, different voice

V-projection drives 50-80% of vocabulary change per head.
Q-projection drives only 5-15%, despite larger weight norms.
LoRA keeps attention patterns similar but rotates what each head outputs.

### 4. Head specialization is learned

Random LoRAs (same architecture, random init) show no consistent head preferences.
The head routing pattern is a property of training on specific text, not architecture.

### 5. Multi-head steering is author-specific

No single head scaling config works for all authors:
- Poe wants H14x1.5 + H5x1.5
- Grimm wants H14x1.5 + H1x1.5
- Shelley wants H14x1.5 + H6x1.5

## Model

**TinyStories-1Layer-21M** (GPT-Neo architecture)

| Property | Value |
|----------|-------|
| Layers | 1 |
| Hidden dim | 1024 |
| Attention heads | 16 (64 dim each) |
| Vocabulary | 50,257 |
| LoRA rank | 8 (on q_proj + v_proj) |
| Trainable params | 32,768 per adapter |

## Project Structure

```
src/sixteen_voices/     # Importable package
  constants.py          # Model constants
  model.py              # Model loading
  adapter.py            # LoRA weight manipulation, knockout, SVD
  steering.py           # Attention head steering via hooks
  text.py               # Prose extraction, perplexity
  dataset.py            # Text chunking for training

scripts/                # Experiment entry points
  knockout.py           # Core head importance experiment
  steer.py              # Multi-head steering
  train_lora.py         # Single-author training
  train_all.py          # Batch training
  transplant.py         # Cross-author head transplant
  vocab_knockout.py     # Per-head vocabulary attribution

data/                   # Download scripts (texts not in repo)
  get_books.py          # 122 books from Gutenberg
  get_author_datasets.py # Combine into 69 author corpora

tests/                  # Unit tests (no model download needed)
```

## Requirements

- Python 3.10+
- PyTorch, Transformers, PEFT, safetensors
- CPU only — all experiments run in minutes to hours

```bash
pip install -e ".[all]"   # includes matplotlib, pytest, ruff
```

## Citation

```bibtex
@misc{sixteenvoices2026,
  title   = {Sixteen Voices: Attention Head Specialization in LoRA-Adapted Language Models},
  year    = {2026},
  url     = {https://github.com/TODO/sixteen-voices},
}
```

## License

MIT
