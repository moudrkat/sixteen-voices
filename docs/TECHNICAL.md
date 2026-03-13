# Technical Report: Sixteen Voices

Detailed description of every experiment in the project, with code
references, equations, and implementation notes. Read
[ARTICLE.md](ARTICLE.md) for the narrative version.

---

## Table of Contents

1. [Key equations](#key-equations)
2. [Training LoRA adapters](#1-training-lora-adapters)
3. [Evaluating adapters](#2-evaluating-adapters)
4. [Per-head knockout](#3-per-head-knockout)
5. [Null baseline (random LoRAs)](#4-null-baseline-random-loras)
6. [Multi-head steering](#5-multi-head-steering)
7. [Head transplant](#6-head-transplant)
8. [Text sample generation](#7-text-sample-generation)
9. [Shared infrastructure](#8-shared-infrastructure)

---

## Key equations

### Attention (single head)

Standard scaled dot-product attention for head *h*:

```
Attn_h(X) = softmax( (X · W_Q_h) · (X · W_K_h)^T / √d ) · (X · W_V_h)
```

where `d = 64` (head dimension), and `W_Q_h`, `W_K_h`, `W_V_h` are the
weight slices for head *h*. The 16 head outputs are concatenated and
projected:

```
MultiHead(X) = Concat(Attn_0, ..., Attn_15) · W_O
```

In this 1-layer model, `MultiHead(X)` feeds directly into the output
(via layer norm + unembedding). There is no second layer — every head
writes directly to the residual stream that produces logits.

### LoRA forward pass

For a frozen weight matrix `W` (e.g., `W_Q` or `W_V`):

```
h = W · x + α/r · B · A · x
```

where:
- `A ∈ R^{r×d_in}` (r=8, d_in=1024) — the down-projection
- `B ∈ R^{d_out×r}` (d_out=1024, r=8) — the up-projection
- `α = 32`, `r = 8` — so the scaling factor is `α/r = 4`

The effective weight change during forward is `α/r · B · A`. However,
in the code, we work with the **un-scaled** product:

```
ΔW = B · A    ∈ R^{1024×1024}    (what the code calls "delta")
```

PEFT applies the `α/r = 4` scaling internally during every forward pass.
All our manipulations (knockout, transplant) operate on the un-scaled
`B·A` — the scaling cancels out in all comparisons since it's applied
consistently by PEFT.

### Head slicing of ΔW

Since `W_Q` and `W_V` are `[1024 × 1024]` matrices where each
64-row block corresponds to one head:

```
ΔW_h = ΔW[h·64 : (h+1)·64, :]    ∈ R^{64×1024}
```

This gives each head's contribution to the adaptation. The full ΔW is
the vertical concatenation of all 16 head slices.

### Knockout (per-head isolation)

To isolate head *h*, construct a modified delta:

```
ΔW_knockout(h) = { ΔW[i,:] if h·64 ≤ i < (h+1)·64
                  { 0       otherwise
```

This matrix is then re-factorized into LoRA matrices via truncated SVD:

```
ΔW_knockout = U · Σ · V^T    (full SVD)
B_new = U[:, :r] · diag(Σ[:r])
A_new = V^T[:r, :]
```

This is the best rank-*r* approximation. Since only 64 of 1024 rows are
nonzero, and those rows came from a rank-8 matrix (the original ΔW =
B·A has rank ≤ 8), the knocked-out delta also has rank ≤ 8. The
truncated SVD at rank 8 is therefore **exact** — no information is lost.

### Recovery score

```
recovery_h = (PPL_base - PPL_h) / (PPL_base - PPL_full)
```

where:
- `PPL_base` = perplexity with frozen model (no adapter)
- `PPL_full` = perplexity with the complete adapter (all 16 heads)
- `PPL_h` = perplexity with only head *h*'s LoRA contribution

### Perplexity

```
PPL = exp( -1/N · Σ_i log P(x_i | x_{<i}) )
```

In practice, PyTorch computes cross-entropy loss averaged over tokens,
so `PPL = exp(loss)`.

### Steering (activation-space intervention)

At inference time, scale head *h*'s output by factor *s*:

```
MultiHead_steered(X) = Concat(s_0·Attn_0, s_1·Attn_1, ..., s_15·Attn_15) · W_O
```

where `s_h = 1.0` for unmodified heads and e.g. `s_h = 2.0` for
amplification or `s_h = 0.0` for silencing.

### Transplant (weight-space intervention)

Replace head *h*'s rows in the recipient's ΔW with the donor's:

```
ΔW_transplant[h·64:(h+1)·64, :] = ΔW_donor[h·64:(h+1)·64, :]
ΔW_transplant[elsewhere]         = ΔW_recipient[elsewhere]
```

Then re-factorize `ΔW_transplant` into A, B via SVD (same procedure as
knockout). **However**, unlike knockout, the transplanted delta mixes
rows from two different rank-8 matrices (donor and recipient), so its
rank can be up to 16. The rank-8 truncated SVD introduces approximation
error here — it is NOT lossless. In practice the error is small (the
transplanted head contributes only 64 of 1024 rows), but this is a
methodological imperfection worth noting.

---

## 1. Training LoRA adapters

**Scripts:** `scripts/train_lora.py` (single author), `scripts/train_all.py` (batch)

**What it does:**
For each of 82 authors, we fine-tune a LoRA adapter on
TinyStories-1Layer-21M. The base model is frozen — only the LoRA
matrices are trained.

**LoRA configuration** (defined in `src/sixteen_voices/constants.py`):

| Parameter | Value | Why |
|-----------|-------|-----|
| Rank | 8 | Small enough to decompose per-head, large enough to learn |
| Alpha | 32 | Standard 4× rank scaling |
| Target modules | `q_proj`, `v_proj` | Q controls what each head attends to, V controls what information it extracts. K is skipped — it's redundant with Q for our analysis (Q·K^T is symmetric in contribution) |
| Dropout | 0.0 | No regularization needed at this scale |

**How it works:**

1. Text is loaded and cleaned via `clean_text()` (`src/sixteen_voices/text.py:8`).
   This removes Gutenberg boilerplate, `[Illustration]` tags, TOC blocks,
   frontmatter (title pages, publisher info), and footnote markers.

2. Cleaned text is tokenized into overlapping chunks via `TextChunkDataset`
   (`src/sixteen_voices/dataset.py:7`):
   - `max_length=512` tokens per chunk
   - `stride=256` — 50% overlap between consecutive chunks
   - Final chunk is always included even if the stride doesn't land on it
   - Labels = input_ids (standard causal LM: predict next token)

3. 90/10 train/val split (sequential, not shuffled — val is the last 10%
   of chunks).

4. Training loop (`train_lora.py:20`):
   - Optimizer: AdamW, lr=5e-4
   - Batch size: 4
   - Epochs: 8
   - No scheduler, no gradient clipping
   - Validation loss computed at end of each epoch

5. Output: adapter saved via `model.save_pretrained()` (PEFT format) to
   `outputs/authors/{name}/adapter/`. Loss curves saved to
   `outputs/authors/{name}/loss.json`.

**Trainable parameters per adapter:**
- A matrix: 8 × 1024 = 8,192 per projection
- B matrix: 1024 × 8 = 8,192 per projection
- Two projections (Q, V): 8,192 × 2 × 2 = 32,768 total
- That's 0.15% of the model's 21M parameters

**Why Q and V, not K?**
In attention, the query Q and key K jointly determine *what* each head
attends to (via Q·K^T), while the value V determines *what information*
flows through. LoRA on Q changes the attention pattern; LoRA on V
changes what gets extracted from attended positions. K is skipped because
adapting Q already gives full control over the attention pattern — Q and
K contribute symmetrically to the attention score.

---

## 2. Evaluating adapters

**Script:** `scripts/eval_adapters.py`

**What it does:**
For each of the 82 adapters, measures whether the adapter actually
learned something — i.e., whether perplexity on the author's own text
decreased compared to the base model.

**How it works:**

1. For each author, extract ~2000 words of prose via `extract_prose()`
   (`src/sixteen_voices/text.py:106`). This function skips past TOC,
   headers, and frontmatter by finding the first line with 60+ characters
   that is >50% lowercase (real prose, not chapter headings).

2. Compute perplexity on that text with the base model and the adapted
   model (`compute_perplexity()`, `text.py:124`):
   - Tokenize with `truncation=True, max_length=512`
   - Forward pass with `labels=input_ids`
   - PPL = exp(cross-entropy loss)

3. Compute ratio = adapted_ppl / base_ppl. An adapter "learned" if
   ratio < 0.85 (i.e., >15% PPL reduction).

**Key detail:** `extract_prose()` is used here and in knockout — this
was a bug fix. Originally `clean_text()` was used for eval, but
`extract_prose()` is better because it skips directly to real prose,
avoiding any residual metadata that survived cleaning.

**Result:** 82/82 adapters showed meaningful learning (ratio < 0.85).

---

## 3. Per-head knockout

**Script:** `scripts/knockout.py`

This is the core experiment of the project.

**What it does:**
For each author × each head (82 × 16 = 1,312 combinations), isolate
that single head's LoRA contribution and measure how much of the full
adapter's improvement it recovers.

**How it works, step by step:**

### Step 1: Load the full adapter and compute ΔW

```
ΔW = B @ A    (shape: 1024 × 1024)
```

This is done by `load_adapter_deltas()` (`src/sixteen_voices/adapter.py:14`).
It loads the safetensors file, extracts A and B for both q_proj and
v_proj, and computes the product.

### Step 2: Slice ΔW by head

The 1024-row ΔW matrix maps directly to 16 heads × 64 dims each:
- Rows 0–63 → Head 0
- Rows 64–127 → Head 1
- ...
- Rows 960–1023 → Head 15

This works because in GPT-Neo's multi-head attention, the Q/V projection
matrix is a single [1024 × 1024] matrix where each 64-row block
corresponds to one head. The outputs are later split and processed
per-head.

### Step 3: Knockout — keep one head, zero the rest

`knockout_all_except(delta, keep_head)` (`adapter.py:28`):
- Create a zero tensor of the same shape
- Copy only the `keep_head` rows (64 rows) from the original delta
- Everything else stays zero

### Step 4: Re-factorize into valid LoRA matrices

The knocked-out ΔW is still rank ≤ 8, but the original A and B matrices
no longer represent it (they encode the full delta, not the zeroed-out
version). We need new A, B. `delta_to_AB()` (`adapter.py:37`) recovers
them via truncated SVD:

```python
U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
B = U[:, :rank] * S[:rank]    # shape: 1024 × 8
A = Vh[:rank, :]               # shape: 8 × 1024
```

This finds the best rank-8 approximation of the knocked-out delta.
Since most rows are zero, this is essentially lossless — the non-zero
block has at most 64 rows and rank ≤ 8.

### Step 5: Inject and measure

`inject_knockout()` (`adapter.py:45`) writes the new A, B directly
into the PeftModel's parameters:

```python
attn.q_proj.lora_A["default"].weight.data.copy_(A_new)
attn.q_proj.lora_B["default"].weight.data.copy_(B_new)
```

Then compute perplexity on the author's text.

### Step 6: Compute recovery score

```
recovery = (base_ppl - knockout_ppl) / (base_ppl - full_ppl)
```

Where:
- `base_ppl` = perplexity with no adapter (frozen model)
- `full_ppl` = perplexity with the complete adapter
- `knockout_ppl` = perplexity with only one head's LoRA active

Interpretation:
- **recovery = 1.0** → this head alone recovers ALL of the adapter's
  improvement
- **recovery = 0.0** → this head contributes nothing (PPL same as base)
- **recovery < 0** → this head alone makes things WORSE than no adapter
  (its contribution, in isolation, is harmful)
- **recovery > 1.0** → this head alone does BETTER than the full adapter
  (other heads were dragging it down)

### Important caveat: recovery scores sum > 1.0

For a single author, summing all 16 recovery scores gives a median of
2.17 across all 82 authors. This is because heads interact — isolating
one head removes interference from others, overstating its contribution
by roughly 2×. The inflation is fairly uniform across heads, so the
**ranking** is preserved: H11 carries 23% after normalization, bottom
tier (H6, H9, H12) stays at 1–2%.

### Key results

- **H11:** Mean recovery 0.338, best head for 41/82 authors. Likely
  carries general coherence.
- **H14:** Most variable (std 0.486). Range from +0.82 (Browne) to
  −1.39 (Burnett). Polarizing — strongly positive for ornate/archaic
  writers, strongly negative for colloquial/folk tales.
- **H3:** Mild generalist (mean 0.30).
- **H6, H7, H12:** Near zero contribution.

---

## 4. Null baseline (random LoRAs)

**Script:** `scripts/knockout_null.py`

**What it does:**
Creates random (untrained) LoRA adapters and runs the same knockout
analysis. Establishes that the head specialization pattern is learned,
not an artifact of random matrix geometry.

**Why this is needed:**
A random rank-8 matrix, when sliced into 16 head blocks, will naturally
have unequal Frobenius norms across heads — just by chance. So "head X
has more weight than head Y" could be meaningless. We need to show that
trained adapters have a *consistent* pattern that random ones don't.

**How it works:**

1. Create a fresh LoRA model via `create_lora_model()` (`model.py:34`).

2. PEFT initializes B=0 (LoRA starts as no-op). Override with random
   weights matching trained scale:
   ```python
   torch.nn.init.normal_(A, std=0.02)
   torch.nn.init.normal_(B, std=0.02)
   ```

3. Run the same knockout procedure (steps 1–6 above) using 6 test
   authors' evaluation texts (Shelley, Grimm, Homer, Poe, Carroll,
   Alcott).

4. Repeat for 5 random seeds.

5. Compare: for each author, is the "best head" consistent across seeds?

**Key result:**
Random adapters give a different "best head" for every seed (Shelley:
H6, H1, H11, H14, H2 across 5 seeds). Trained adapters converge:
41/82 agree on H11 as best. That convergence is learned.

**Limitation acknowledged in the article:** We verified consistency
*across authors* (many authors agree on H11), not across multiple
training runs of the *same* author. The latter would be a stronger test.

---

## 5. Multi-head steering

**Script:** `scripts/steer.py`

**What it does:**
Instead of modifying LoRA weights (knockout = weight-space
intervention), steering modifies head outputs at inference time via
forward hooks. Scale a head's output by 2× to amplify it, or by 0× to
silence it.

**How it works:**

1. Register a forward hook on the attention output projection
   (`model.py:60`, `get_attn_out()` returns `attn.out_proj`).

2. The hook (`steering.py:8`, `make_hook()`) intercepts the attention
   output tensor (shape: `[batch, seq_len, 1024]`) and scales specific
   64-dim slices:
   ```python
   for head_idx, scale in head_scales.items():
       s = head_idx * HEAD_DIM
       h[:, :, s : s + HEAD_DIM] *= scale
   ```

3. This is applied during both generation (for text samples) and
   perplexity computation (for quantitative evaluation).

**Configurations tested per author** (defined in `steer.py:33`):
- `baseline` — no scaling (reference)
- `H14_x2` — double H14's output
- `H14_x0` — silence H14 completely
- `H{best}_x2` — double the author's best non-H14, non-H11 head
- Various combinations: `H14x2+H{best}x2`, `H14x2+H{best}x1.5`, etc.
- `top3_x1.5` — mild amplification of H3, H11, H14 together

**Difference from knockout:**
- Knockout operates on **weights** (ΔW). It modifies the LoRA matrices
  permanently. The intervention happens before attention is computed.
- Steering operates on **activations**. The LoRA weights are unchanged;
  only the output of each head is scaled after attention is computed.
- Knockout is per-head isolation (keep one, zero rest). Steering is
  per-head amplification/suppression (scale factors on top of the full
  adapter).

**Output:** `outputs/multihead_text.json` — PPL and generated text for
each author × configuration.

---

## 6. Head transplant

**Script:** `scripts/transplant.py`

**What it does:**
Take one author's ΔW rows for a specific head and graft them into
another author's adapter. If H14 carries Poe's "darkness vocabulary,"
transplanting Poe's H14 into Carroll's adapter should shift Carroll's
output toward darker words while keeping Carroll's sentence structure.

**How it works:**

1. Load both adapters' deltas via `load_adapter_deltas()`.

2. `transplant_heads()` (`transplant.py:43`):
   ```python
   result = recipient_delta.clone()
   for h in heads:
       start = h * HEAD_DIM
       end = start + HEAD_DIM
       result[start:end, :] = donor_delta[start:end, :]
   ```
   Simply copy the donor's 64 rows into the recipient's delta matrix.

3. Re-factorize the hybrid delta back into LoRA A, B via `delta_to_AB()`
   and inject into the model via `inject_deltas()` (`transplant.py:53`).

4. Generate text with the same prompts and seed as the pure adapters.
   Compute perplexity on both authors' texts.

**The article uses three pairs** (all with donor=Poe, head=14):
- Poe → Carroll
- Poe → Grimm
- Poe → Minimalist (synthetic control)

These were selected because they produce readable output — most
donor-host pairs produce garbage.

**What makes transplant different from steering:**
- Steering scales existing head outputs multiplicatively. It can
  amplify or suppress what a head already does.
- Transplant replaces head weights entirely. The recipient gets the
  donor's learned attention pattern and value extraction for that head.
  This is a structural change, not just a gain adjustment.

**Reproducibility:** The transplant figure uses outputs from
`outputs/transplant_samples.json`, generated with prompt="It was a dark
and stormy", seed=42. The figure script (`scripts/fig_transplant.py`)
reads this JSON directly.

---

## 7. Text sample generation

**Script:** `scripts/generate_samples.py`

**What it does:**
Generates text from the base model and all 82 adapted models using a
fixed prompt and seed, saving exact outputs to JSON for reproducibility.

**How it works:**

1. Load tokenizer, then for each model (base + 82 adapters):
   - Load model via `load_adapted_model()` or `AutoModelForCausalLM`
   - Call `generate()` (`steering.py:28`) with:
     - `prompt="It was a dark and stormy"`
     - `seed=42`
     - `max_new_tokens=60`
     - `temperature=0.8`
     - `top_k=50`

2. Generation uses `torch.manual_seed(seed)` before each call for
   reproducibility.

3. Output saved to `outputs/samples.json` with metadata (prompt, seed,
   max_new_tokens) and all 83 samples.

**The article cherry-picks 12 samples** (base + 11 authors) from this
JSON to show stylistic contrast. The selection criteria: visually
distinct outputs that illustrate different aspects of adaptation
(dialogue, verse, gothic, minimalist, etc.).

---

## 8. Shared infrastructure

### Text processing pipeline

Two separate functions serve different purposes:

**`clean_text()`** (`text.py:8`) — used for **training**:
- Removes `[Illustration]` tags, `[1]` footnote markers
- Strips `=== Book Title ===` separators
- Removes TOC blocks (CONTENTS heading + short lines)
- Removes frontmatter (5+ consecutive short non-prose lines)
- Safety: if stripping would remove >50% of lines, skip it (probably
  poetry)
- Returns full cleaned text

**`extract_prose()`** (`text.py:106`) — used for **evaluation**:
- Finds first line with 60+ chars that is >50% lowercase
- Returns `length` characters starting from that point (default 5000)
- Faster and more targeted than `clean_text()` — skips directly to
  prose without trying to clean everything

### Perplexity computation

`compute_perplexity()` (`text.py:124`):
- Tokenizes with `truncation=True, max_length=512`
- Single forward pass with `labels=input_ids`
- Returns `exp(loss)` — the geometric mean of per-token probabilities

Note: this only measures perplexity on the first 512 tokens. For longer
eval texts, only the beginning is scored. This is consistent across all
experiments (same function everywhere).

### Model loading

Three entry points (`model.py`):
- `load_base_model()` — frozen TinyStories-1Layer-21M
- `load_adapted_model(path)` — base + PEFT adapter from disk
- `create_lora_model()` — base + fresh (untrained) LoRA config

All use `roneneldan/TinyStories-1Layer-21M` from HuggingFace.

### Constants

Defined once in `src/sixteen_voices/constants.py`:
```python
MODEL_NAME = "roneneldan/TinyStories-1Layer-21M"
NUM_HEADS = 16
HEAD_DIM = 64
HIDDEN_DIM = 1024
RANK = 8
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "v_proj"]
```

---

## Experiment dependency graph

```
train_lora.py (×82)
    ↓
    outputs/authors/{name}/adapter/
    ↓
    ├── eval_adapters.py → outputs/eval_adapters.json
    ├── knockout.py → outputs/knockout_all_heads.json
    │       ↓
    │       ├── fig_knockout_heatmap.py → figures/knockout_heatmap.png
    │       │                           → figures/knockout_strip.png
    │       └── knockout_null.py → outputs/knockout_null_baseline.json
    ├── steer.py → outputs/multihead_text.json
    ├── transplant.py → outputs/transplant_samples.json
    │       ↓
    │       └── fig_transplant.py → figures/transplant.png
    └── generate_samples.py → outputs/samples.json
```

All experiments depend on trained adapters. Knockout depends on base
model perplexities. Null baseline depends on knockout results for
comparison. Figure scripts depend on experiment JSON outputs.