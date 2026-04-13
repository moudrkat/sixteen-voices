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
7. [Steering sweep](#6-steering-sweep)
8. [Attention pattern analysis](#7-attention-pattern-analysis)
9. [Head probes](#8-head-probes)
10. [Author-specific style heads](#9-author-specific-style-heads)
11. [H14 V-Q balance](#10-h14-v-q-balance)
12. [Why H14? Base model weight analysis](#10a-why-h14-base-model-weight-analysis)
13. [Head role prediction](#10a-ii-head-role-prediction-from-base-model-weights)
14. [Retraining stability](#10b-retraining-stability)
15. [Attention pattern stability](#10c-attention-pattern-stability-across-adapters)
16. [V-only vs Q-only knockout](#10d-v-only-vs-q-only-knockout)
17. [Head transplant](#11-head-transplant)
18. [Text sample generation](#12-text-sample-generation)
19. [Shared infrastructure](#13-shared-infrastructure)

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

At inference time, scale head *h*'s output by factor *s* **before** the
W_O projection (i.e. on the concatenated head outputs, not after mixing):

```
MultiHead_steered(X) = Concat(s_0·Attn_0, s_1·Attn_1, ..., s_15·Attn_15) · W_O
```

where `s_h = 1.0` for unmodified heads and e.g. `s_h = 2.0` for
amplification or `s_h = 0.0` for silencing. In code, this is implemented
as a `register_forward_pre_hook` on `out_proj`, which intercepts the
input to W_O before projection.

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
For each of 77 authors, we fine-tune a LoRA adapter on
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
For each of the 77 adapters, measures whether the adapter actually
learned something — i.e., whether perplexity on the author's own text
decreased compared to the base model.

**How it works:**

1. For each author, extract ~2000 words of prose via `extract_prose()`
   (`src/sixteen_voices/text.py:106`). This function skips past TOC,
   headers, and frontmatter by finding the first line with 40+ characters
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

**Result:** 77/77 adapters showed meaningful learning (ratio < 0.85).

---

## 3. Per-head knockout

**Script:** `scripts/knockout.py`

This is the core experiment of the project.

**What it does:**
For each author × each head (77 × 16 = 1,232 combinations), isolate
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
2.17 across all 77 authors. This is because heads interact — isolating
one head removes interference from others, overstating its contribution
by roughly 2×. The inflation is fairly uniform across heads, so the
**ranking** is preserved: H11 carries 23% after normalization, bottom
tier (H6, H9, H12) stays at 1–2%.

### Key results

- **H11:** Mean recovery 0.384, best head for 51/77 authors (66%).
  The dominant head for most authors.
- **H3:** Mean recovery 0.284 — second by mean, consistently useful
  across nearly all authors, but rarely the single best (1/77).
  Correlated with H14 (r = 0.72) — they serve the same authors.
- **H14:** Mean recovery 0.221, highest variance (std 0.291). Best
  head for 18/77 authors (23%) — a specific cluster: Homer (+0.73),
  Melville (+0.71), Egyptian (+0.71), Maya (+0.71), Milton (+0.68),
  Browne (+0.69), Carlyle (+0.67), Lovecraft (+0.62). Actively hurts
  for 9 authors: Shelley (−0.68), Stoker (−0.35), Wells (−0.34),
  Wilde (−0.34), Baum (−0.28), Korean (−0.28), Kipling (−0.28),
  Pyle (−0.22), Twain (−0.10).
- **H6, H12:** Near zero contribution.
- H11 and H14 are anticorrelated across authors (r = −0.39).
  When H14 leads, it recovers more on average (0.62) than H11 does
  when it leads (0.46). They do the same job — main style carrier —
  for different author groups. See [ARTICLE_SIMPLE.md](ARTICLE_SIMPLE.md)
  for the two-strategy analysis.

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
51/77 agree on H11 as best. That convergence is learned.

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

## 6. Steering sweep

**Script:** `scripts/steering_sweep.py`

**What it does:**
For each of 77 authors, scales their most important heads from 0× to
2× in steps of 0.25 and measures PPL at each step. Produces "steering
curves" — how smoothly each author responds to continuous head scaling.

**How it works:**

1. Load knockout data to rank heads per author.

2. For each author, select heads to sweep (default: top 2 by absolute
   recovery + H11 + H14 — always interesting).

3. For each head × scale factor (9 points: 0.0, 0.25, ..., 2.0):
   - Register a forward hook via `make_hook({head: scale})`
   - Compute PPL on author's eval text
   - Remove hook

4. All other heads remain at 1× — only the target head is scaled.

**Scale factors tested:** `[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]`

**Key results:**

- **224 steering curves** total (77 authors × ~3 heads each)
- **H11** has symmetric V-shaped curves — killing it hurts, amplifying
  it hurts, everyone needs it at ~1×. Universal coherence head.
- **H14** has asymmetric, author-dependent curves:
  - For H14-dominant authors (Homer, Melville, Milton): killing hurts
    more than amplifying. These authors need H14's contribution.
  - For H11-dominant authors: H14 curves are flat — scaling it
    barely affects PPL. H14 simply isn't doing much for them.
- The steering aggregate figure (`figures/steering_aggregate.png`) shows
  all 77 curves overlaid — H11 is a clean fan, H14 fans out for the
  specialist cluster while the majority stays flat.

**Figure script:** `scripts/fig_steering.py`

**Output:** `outputs/steering_sweep.json`

---

## 7. Attention pattern analysis

**Script:** `scripts/head_attention_patterns.py`

**What it does:**
Analyzes what each attention head attends to — the classic Voita et al.
interpretability approach. Measures position bias, token type preference,
and attention entropy for each head. Runs on both the base model and
adapted models.

**How it works:**

1. Run 10 diverse sentences through the model with `output_attentions=True`.

2. For each head, compute statistics across all positions in all texts:
   - **Previous-token fraction:** how much attention goes to position `i-1`
   - **First-token fraction:** how much goes to position 0 (BOS)
   - **Local fraction:** how much goes to positions within 3 tokens
   - **Entropy:** `−Σ p·log₂(p)` — low = focused, high = diffuse
   - **Function-word fraction:** attention to "the", "and", "was", etc.
   - **Punctuation fraction:** attention to `.`, `,`, `!`, etc.

3. Classify each head into a pattern type:
   - `previous-token` — strong previous-token attention (>0.3)
   - `local-window` — most attention within 3 tokens (>0.7)
   - `focused` — low entropy (<1.5)
   - `diffuse` — high entropy (>3.0)
   - `function-words` — high function word attention (>0.5)
   - `mixed` — no dominant pattern

**Key results (base model):**

| Head | Entropy | Prev-token | Local | Pattern |
|------|---------|------------|-------|---------|
| H4   | 1.34    | 0.33       | 0.95  | previous-token, local-window, focused |
| H6   | 1.47    | 0.43       | 0.96  | previous-token, local-window, focused |
| H8   | 1.27    | 0.14       | 0.92  | local-window, focused |
| H12  | 1.13    | 0.32       | 0.86  | previous-token, local-window, focused |
| H11  | 2.64    | 0.16       | 0.54  | mixed (semantic) |
| H14  | 2.31    | 0.17       | 0.53  | mixed (semantic) |

Three heads (H4, H6, H12) are clearly **previous-token/induction heads**
— focused, local, attending strongly to the preceding position. H8 is
a local n-gram head without the prev-token peak.

Everything else is **semantic** — diffuse attention spread across broad
context.

**Critical finding:** Attention patterns barely change between base and
adapted models. LoRA adapters modify *what heads output*, not *what they
attend to*. Style lives in the value projections, not the routing.

This explains the knockout results: structural heads (H4, H6, H8, H12)
have low knockout recovery because their fixed copying behavior doesn't
change per author. Semantic heads (H11, H14, etc.) carry the style
because their broad attention is processed through modified V projections.

**Output:** `outputs/head_attention_patterns.json`

**Figure:** `figures/head_roles.png` (combined with knockout data)

---

## 8. Head probes

**Script:** `scripts/head_probes.py`

**What it does:**
For each head, generates text with three conditions — head solo (only
that head at 1×, all others at 0×), head killed (that head at 0×),
and head amplified (that head at 2×). Runs on both the base model and
adapted models to compare head roles with and without LoRA.

**How it works:**

1. For each of 16 heads:
   - **Solo:** scale all heads to 0× except this one (at 1×)
   - **Kill:** scale only this head to 0× (all others at 1×)
   - **Amplify:** scale this head to 2× (all others at 1×)

2. For each condition: compute PPL on a standard eval text and generate
   text from 2 prompts.

3. Compute `ppl_impact = kill_ppl − normal_ppl` — positive means this
   head helps (killing it hurts).

4. Word frequency analysis: compare word distributions between amplified
   and normal text to find "boosted" words.

**Key results (base model vs adapted):**

Base model: all heads have near-zero impact when killed (±1 PPL). No
single head is critical — the base model is robust to single head loss.

Adapted models: impact varies dramatically per head. For Poe, killing
H8 costs +6 PPL and H12 costs +8 PPL. For Browne, killing H8 costs +9
PPL. These are the heads where the LoRA adapter concentrated its
changes.

Solo generation (base model, one head active) produces mostly garbage
— repetitive tokens, incoherent fragments. This is expected: a 1-layer
model needs all heads cooperating.

**Output:** `outputs/head_probes.json`

---

## 9. Multi-head steering experiments

**Scripts:** `scripts/fig_poe_steering.py`, `scripts/fig_poe_multihead.py`,
`scripts/fig_poe_style_dial.py`, `scripts/poe_steering_prompts.py`

**What these do:**
Several experiments explored whether steering (scaling head activations
at inference time) produces visibly different text. This complements
the PPL-based steering sweep (section 6) with qualitative text samples.

**Experiments run:**

1. **Single-head steering** (`poe_steering_prompts.py`): Scale Poe's
   H14 from 0× to 2× across 15 prompts. Effect on generated text is
   subtle — hard to see a clear style shift in individual samples.

2. **Top-3 multi-head** (`fig_poe_multihead.py`): Scale H14+H3+H5
   together. More dramatic than single-head, but still noisy.

3. **Author-specific heads** (`fig_poe_style_dial.py`): Select heads
   with knockout recovery > 0.2 for Poe (H2, H3, H5, H13, H14, H15 —
   6 heads). Scale all 6 together from 0.25× to 1.5×, keeping others
   at 1×. Wider usable range than scaling all semantic heads (which
   breaks the model at both extremes because H11 carries coherence,
   not Poe-specific style).

**Honest assessment:**
The PPL steering curves (section 6) show clean, measurable effects —
H11 and H14 behave very differently across 77 authors. The generated
text samples are less convincing: at low scales the model degrades
(repetitive, incoherent), and at high scales it also degrades (loses
topic). The 21M-parameter model doesn't have enough capacity for
steering to produce a clean "more style / less style" gradient in
generated text the way it shows up in perplexity measurements.

The quantitative result (PPL curves) is solid. The qualitative result
(text samples) is noisy and shouldn't be oversold.

**Caveat:** The recovery > 0.2 threshold for selecting "author-specific
heads" is arbitrary. The heads selected are specific to this checkpoint.

**Output:** `outputs/poe_style_dial_samples.json`,
`outputs/poe_steering_prompts.json`, `outputs/poe_multihead_samples.json`

---

## 10. H14 V-Q balance

**Scripts:** `scripts/h14_vq_balance.py`, `scripts/h14_correlates.py`,
`scripts/h14_vocab_vs_structure.py`, `scripts/h14_attention_features.py`

**What it does:**
Investigates what separates H14-dominant authors from H11-dominant
authors. Tests text-level metrics and LoRA weight structure as
predictors.

**Text-level correlates** (`h14_correlates.py`):

Computed on cleaned text (same preprocessing as training):

| Metric | Correlation with H14 recovery |
|--------|-------------------------------|
| avg_word_len | +0.42 |
| pct_short_words | −0.42 |
| simple_word_frac | −0.41 |
| pct_long_words | +0.40 |
| latinate_frac | +0.35 |
| type_token_ratio | +0.30 |
| comma_density | +0.27 |
| base_ppl | +0.09 (not significant) |

Vocabulary complexity partially predicts H14 recovery, but only
explains ~18% of variance. Base model perplexity (distributional
distance) has essentially zero correlation (r = −0.04).

**V-Q balance** (`h14_vq_balance.py`):

For each author, compute the fraction of H14's LoRA weight norm in
the value projection vs the query projection:

```
V-Q balance = ||ΔW_V[H14]|| / ||ΔW_V|| − ||ΔW_Q[H14]|| / ||ΔW_Q||
```

This single metric correlates with H14 recovery at **r = +0.62
(R² = 0.38)**.

**Interpretation (plausible but not directly tested via this
experiment alone):** LoRA adapts both Q (what the head attends to)
and V (what it extracts from attended positions). V changes are
local to the head — they modify output regardless of other heads.
Q changes are contextual — the new routing was learned with all
heads active and may break in isolation.

Note: This knockout experiment keeps BOTH Q and V LoRA for the
isolated head. To directly test whether V works better than Q in
isolation, a separate V-only vs Q-only knockout is needed (see
section 10d).

**Specificity to H14:**
The V-Q balance has strong predictive power only for H14:

| Head | r(V-Q, recovery) | recovery std |
|------|-------------------|--------------|
| H14  | +0.62             | 0.29         |
| H1   | +0.37             | 0.19         |
| H15  | +0.16             | 0.21         |
| H4   | −0.32             | 0.12         |
| Others | |r| < 0.2       | 0.09–0.27    |

H14 is the head where both recovery variance and V-Q predictive
power are highest.

**Synthetic control validation:**

The synthetic authors confirm the V-Q interpretation:

| Style type | Authors | V-Q balance | H14 |
|---|---|---|---|
| Vocabulary-defined | unusual_vocab, reporter, simple_vocab | least negative | positive |
| Mixed | poet, dark, cozy | middle | mixed |
| Structure-defined | dialogue, firstperson, repeater, fabulist, ... | most negative | neutral/negative |

Vocabulary-defined styles (word choice only) → V-heavy → works in
isolation. Structure-defined styles (sentence patterns) → Q-heavy →
breaks in isolation.

**Can text metrics predict V-Q balance?**

Tested three approaches:

| Approach | r with V-Q | r with H14 |
|---|---|---|
| Surface text metrics (word length, vocab) | ~0.3 | ~0.4 |
| Unigram KL + bigram structure composite | +0.29 | +0.16 |
| Shuffle test (model-based, destroy word order) | −0.16 | −0.20 |

Surface text metrics are weak predictors. The V-Q balance captures
something that lives in the weights, not in surface statistics.

**Base model attention as predictor** (`h14_attention_features.py`):

Running the base model (no adapter) on each author's text and
extracting per-head attention entropy:

| Feature | r(V-Q) | r(H14) |
|---|---|---|
| H10 entropy | −0.55 | −0.51 |
| H15 entropy | −0.57 | −0.50 |
| H7 entropy | −0.47 | −0.48 |
| H5 entropy | −0.50 | −0.45 |
| H0 entropy | −0.50 | −0.43 |
| H14 entropy | −0.11 | −0.06 |

Multiple heads' entropy correlates at r = −0.4 to −0.5 with H14
recovery. H14's own entropy is not predictive. The sign means:
authors whose text makes the base model attend more sharply (lower
entropy across many heads) tend to be H14-positive. A multivariate
model using all 32 attention features reaches R² = 0.52 for H14
recovery, but with 33 features / 77 samples this is likely overfit.

The base model's internal representations contain more signal than
surface text statistics, suggesting learned representations (e.g.
hidden state statistics) could serve as input features for a
hypernetwork predicting LoRA weights.

**Output:** `outputs/h14_vq_balance.json`, `outputs/h14_correlates.json`,
`outputs/h14_attention_features.json`, `outputs/h14_vocab_vs_structure.json`,
`figures/h14_vq_balance.png`, `figures/h14_attention_features.png`

---

## 10a. Why H14? Base model weight analysis

**Script:** `scripts/why_h14.py`

**What it does:**
Probes the pretrained base model weights (before any LoRA) to find
what makes H14 structurally different from other heads.

**Tests and results:**

| Test | What it measures | H14 rank | Key finding |
|------|-----------------|----------|-------------|
| V-perturbation sensitivity | Mean |logit change| after adding noise to V | **#1** (0.222, next is H11 at 0.179) | H14's output is most sensitive to V changes |
| Q-perturbation sensitivity | Attention KL divergence after adding noise to Q | **#2** (0.0118, #1 is H9 at 0.0118) | H14's attention pattern is highly sensitive to Q changes |
| V effective rank | Shannon entropy of singular values | **#1** (63.81) | V weights are maximally spread — no single direction dominates |
| V norm | Frobenius norm of V weight block | **#15** (7.56) | Small V weights despite high sensitivity |
| Q norm | Frobenius norm of Q weight block | #5 (8.99) | Average |
| W_O influence | Output projection column norm | #10 (8.44) | Average influence on residual stream |
| Head independence | 1 − mean |correlation| with other heads | **#15** (most correlated) | Most dependent on other heads |

**Perturbation methodology:**
- Random Gaussian noise at scale 0.01 added to each head's Q or V
  weight block (64 × 1024)
- Q sensitivity: measured as KL divergence of attention distribution
  before/after perturbation, averaged over 5 prompts × 10 trials
- V sensitivity: measured as mean |logit difference| at the output,
  averaged over 5 prompts × 10 trials

**Interpretation:**
H14 has small but maximally spread-out V weights. This makes it a
natural amplifier — tiny LoRA changes to V produce outsized effects
on the output. It's also extremely Q-sensitive, so Q changes
redirect its attention dramatically. Combined with being the least
independent head (most correlated with others), Q changes that
redirect H14's attention break without other heads compensating.

This explains the V-Q balance finding: V-heavy LoRA adaptation
works in isolation because it amplifies through H14's sensitive V
pathway. Q-heavy adaptation breaks in isolation because H14's
attention routing is both highly sensitive and highly dependent on
other heads.

**Output:** `outputs/why_h14.json`, `figures/why_h14.png`,
`figures/why_h14_sensitivity.png`

**Figure script:** `scripts/fig_why_h14.py`

---

## 10a-ii. Head role prediction from base model weights

**Scripts:** `scripts/fig_head_mechanics.py`, `scripts/why_h14.py`

**What it does:**
Tests whether base model properties (before any LoRA) predict
which heads will matter after fine-tuning. Extends the perturbation
analysis to explain both head importance and head variance.

**Key finding: two properties predict two things**

| What to predict | Best predictor | r | Mechanism |
|---|---|---|---|
| Mean recovery (importance) | Logit impact (‖W_unembed · W_O_h‖) | +0.78 | Heads with stronger output paths to vocabulary matter more |
| Recovery std (variance) | V-sensitivity | +0.86 | Heads more sensitive to V perturbations vary more across authors |

**The amplification ratio**

V-sensitivity is itself explained by the ratio of logit impact to
V weight norm:

```
amplification_ratio_h = ‖W_unembed · W_O_h‖ / ‖W_V_h‖
```

This predicts V-sensitivity at r = +0.91 (r = +0.86 with H14
removed, r = +0.79 with both H14 and H11 removed — not an outlier
effect).

**Interpretation:** A head with a strong output path (high logit
impact) but small V weights (low V norm) is an amplifier — the
same absolute LoRA change to V produces a larger relative effect
on logits. H14 has the highest amplification ratio (9.07) because
its V weights are the smallest (norm 7.56, rank #15) while its
logit path is well-connected (68.6, rank #5).

**Q-sensitivity is different:**

Q_norm alone predicts Q-sensitivity at r = +0.76. Bigger Q weights
mean larger attention scores (Q·K^T), so perturbations create
larger absolute changes in the softmax. This is a simpler mechanism
than V-sensitivity.

**Per-head summary:**

| Head | Type | Amplification ratio | V-sensitivity | Logit impact | Recovery mean | Recovery std |
|------|------|-------------------|--------------|-------------|--------------|-------------|
| H14 | semantic | **9.07** (#1) | **0.222** (#1) | 68.6 (#5) | 0.22 (#3) | **0.29** (#1) |
| H11 | semantic | 8.60 (#2) | 0.179 (#2) | **100.4** (#1) | **0.38** (#1) | 0.17 (#2) |
| H3  | semantic | 8.02 (#4) | 0.153 (#5) | 67.0 (#6) | 0.28 (#2) | 0.14 (#6) |
| H12 | structural | 7.49 (#11) | 0.126 (#12) | 54.2 (#16) | 0.08 (#15) | 0.08 (#14) |

H11 is #1 in logit impact → workhorse (highest mean, best for 51/77).
H14 is #1 in amplification ratio → wildcard (highest variance, best for 18/77, hurts for 9).
H12 is near-bottom in both → irrelevant.

**Caveat: partial tautology.**
V-sensitivity (random noise perturbation of V) and knockout recovery
(learned LoRA perturbation of V) both measure "how much does
changing V affect the output." Their correlation is partly expected.
The amplification ratio decomposes V-sensitivity into its own
propagation path (logit impact) and relative perturbation size
(1/V_norm), which is closer to redescription than explanation.

The more substantive observation is the *separation*: logit impact
predicts mean recovery (importance) while V-sensitivity predicts
recovery variance. These are different properties
predicting different behavioral outcomes. However, with only N=16
heads, all correlations are suggestive rather than conclusive.

**Output:** `outputs/head_mechanics.json`, `figures/head_mechanics.png`

---

## 10b. Retraining stability

**Script:** `scripts/retrain_stability.py`

**What it does:**
Tests whether knockout rankings are stable across different random
seeds. Retrains Poe's adapter 5 times with seeds 42–46, runs the
full knockout on each, and checks ranking consistency.

**Results (Poe × 5 seeds):**

| Metric | Value |
|--------|-------|
| H14 rank | 3rd every time (std = 0.0) |
| H14 recovery | mean = +0.162, std = 0.003 |
| Top head | H11 every time |
| Full ranking | Preserved across all 5 seeds |

The stability is remarkably high — not just the top head but the
full 16-head ranking is preserved. This confirms that the knockout
results reflect genuine learned structure, not training noise.

**Output:** `outputs/retrain_stability.json`, `figures/retrain_stability.png`

Adapters saved to `outputs/retrain_stability/poe/seed_{N}/adapter/`
for reproducibility.

---

## 10c. Attention pattern stability across adapters

**Script:** `scripts/fig_attention_stability.py`

**Requires:** `outputs/head_attention_patterns.json` (produced by
running `head_attention_patterns.py --with-adapters` on all 77 authors)

**What it does:**
Compares per-head attention entropy between the base model and all 77
adapted models. Tests whether LoRA changes what heads attend to or
only what they output.

**Key results:**

- Mean |entropy change| across all heads × all authors: ~0.02–0.05
  (negligible compared to between-head entropy differences of 1–3)
- Semantic head classifications: **0 changes** across all 77 adapted
  models — every head that was "semantic" in the base model stays
  semantic after LoRA, and vice versa for structural heads
- Structural heads (orange) and semantic heads (purple) stay in place
  after adaptation

This is the strongest evidence that LoRA adapters change *what heads
output* (value projections), not *what they attend to* (query/key
routing). Style flows through V, not Q — even though Q is also
adapted.

**Output:** `outputs/attention_stability.json`, `figures/attention_stability.png`

---

## 10d. V-only vs Q-only knockout

**Script:** `scripts/vq_knockout.py`

**What it does:**
Directly tests the V-Q mechanism by isolating each projection
separately. For H14 across all 77 authors, runs three conditions:

- **V-only:** keep only H14's V LoRA, zero all Q LoRA
- **Q-only:** keep only H14's Q LoRA, zero all V LoRA
- **Both:** keep H14's Q+V LoRA (existing knockout result from §3)

If V changes are truly local (work in isolation) and Q changes are
contextual (break in isolation), V-only should consistently recover
more than Q-only.

**Implementation:**
Uses `load_adapter_deltas()` to get the full V and Q deltas, then
`knockout_all_except()` to isolate H14's rows. Each condition applies
only the V delta or only the Q delta to a fresh copy of the base model.

**Results:**

| Metric | Value |
|--------|-------|
| V-only > Q-only | **68 / 77 authors** (88%) |
| Mean V-only recovery | **+0.09** |
| Mean Q-only recovery | **−0.03** |
| r(V-Q balance, V−Q recovery diff) | **+0.67** |

V-only recovery is positive for most authors — isolating just
H14's value projection still helps. Q-only recovery is negative for
the majority — isolating just the query projection actively hurts.

**Correlation with V-Q balance:**
Authors whose LoRA puts more weight in V (higher V-Q balance) also
show a larger gap between V-only and Q-only recovery (r = +0.67).
This connects the weight-level measurement (§10) with the behavioral
test: V-heavy adapters benefit more from V in isolation.

**Why this matters:**
Before this experiment, the V-Q story rested on a correlation
(V-Q balance vs H14 recovery, r = +0.62) plus a plausible mechanism.
This experiment tests the mechanism directly: V changes literally
work better than Q changes in isolation, for 68/77 authors. The
mechanism is not just an interpretation of the correlation — it's
a separately testable prediction that holds.

**Output:** `outputs/vq_knockout.json`, `figures/vq_knockout.png`

---

## 11. Head transplant

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

## 12. Text sample generation

**Script:** `scripts/generate_samples.py`

**What it does:**
Generates text from the base model and all 77 adapted models using a
fixed prompt and seed, saving exact outputs to JSON for reproducibility.

**How it works:**

1. Load tokenizer, then for each model (base + 77 adapters):
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

## 13. Shared infrastructure

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
- Finds first line with 40+ chars that is >50% lowercase
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
train_lora.py (×77)
    ↓
    outputs/authors/{name}/adapter/
    ↓
    ├── eval_adapters.py → outputs/eval_adapters.json
    ├── knockout.py → outputs/knockout_all_heads.json
    │       ↓
    │       ├── fig_knockout_heatmap.py → figures/knockout_heatmap.png
    │       │                           → figures/knockout_strip.png
    │       ├── knockout_null.py → outputs/knockout_null_baseline.json
    │       └── steering_sweep.py → outputs/steering_sweep.json
    │               ↓
    │               └── fig_steering.py → figures/steering_curves.png
    │                                   → figures/steering_aggregate.png
    ├── head_attention_patterns.py → outputs/head_attention_patterns.json ─┐
    │                                                                      ├→ fig_steering.py → figures/head_roles.png
    │   knockout.py ──────────────────────────────────────────────────────┘
    ├── head_probes.py → outputs/head_probes.json
    ├── fig_poe_style_dial.py → outputs/poe_style_dial_samples.json
    │                         → figures/poe_style_dial_*.png
    ├── steer.py → outputs/multihead_text.json
    ├── transplant.py → outputs/transplant_samples.json
    │       ↓
    │       └── fig_transplant.py → figures/transplant.png
    └── generate_samples.py → outputs/samples.json
```

All experiments depend on trained adapters. Knockout depends on base
model perplexities. Null baseline depends on knockout results for
comparison. Figure scripts depend on experiment JSON outputs.