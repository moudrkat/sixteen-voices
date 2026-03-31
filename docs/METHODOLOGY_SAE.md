# SAE Interpretability Methodology

How to go from "I have a model" to "I know what it computes" — the pipeline we developed on TinyStories-1Layer-21M.

---

## The problem

A trained model produces different outputs for different inputs, but you can't see why. The residual stream is a 1024-dimensional vector where individual dimensions mean nothing. Attention heads matter (knockout experiments tell you which), but not what they compute. You need to decompose the internal representation into interpretable pieces.

## Step 0: Design controls before looking inside

Train synthetic LoRA adapters that each isolate a single style property:

| Synthetic | What it isolates |
|---|---|
| minimalist | Short sentences, simple words |
| dialogue | All conversation, quoted speech |
| firstperson | "I" narration throughout |
| questioner | Interrogative-driven text |
| cozy | Warm domestic settings |
| dark | Atmospheric tension |
| repeater | Rhythmic repetition |
| reporter | Factual, declarative |
| unusual_vocab | Deliberately obscure words |
| poet | Line breaks, verse rhythm |

These exist before the SAE. They are the ground truth.

Most SAE work labels features post-hoc — look at what fires and guess what it means. Here the labels come from designed controls. When a feature correlates with "questioner" and its top tokens are question marks, the label is grounded, not inferred.

## Step 1: Train the SAE

Collect activations from the base model's residual stream on all author texts. Train a sparse autoencoder to decompose them.

**What we learned about sparsity:** Our first SAE (256 features, ReLU, L1 penalty λ=0.001) had 99% firing rate — not sparse at all. It found structure (author-discriminating directions), but features were polysemantic blends.

Switching to **TopK activation** (Gao et al. 2024) — keep only the top k activations per token, zero the rest — produced actual sparsity. With k=16 out of 2048 features, the mean firing rate dropped to 4.3%, with the most selective features at 0.3%.

Key parameters:
- Overcomplete (more features than input dimensions) works when sparsity is enforced
- Dead features are expected — the model tells you how many directions it actually uses
- Explained variance matters — how much of the original signal survives the bottleneck

## Step 2: Label features with controls

For each feature, check three things:

1. **Which tokens fire it?** Run all text through the SAE, find the tokens where each feature activates most strongly.
2. **Which synthetics correlate?** Compute mean feature activation per author. Which designed controls score highest/lowest?
3. **Do they agree?** If a feature fires on question marks AND correlates with the questioner synthetic, it's a question-detection feature. If the tokens say one thing and the synthetic says another, don't label it.

This three-way validation rejects loose abstractions. Our first attempt labeled a feature "complexity" because it was high for Homer and Poe. The tokens didn't support it — the label was an abstraction that felt right but wasn't grounded.

## Step 3: Connect features to heads

Correlate each feature with each head's knockout recovery score (from the previous experiment). This answers: which heads read which features?

Statistical correction: Benjamini-Hochberg (FDR=0.05) for the feature × head matrix. This is ahead of current SAE field practice, which typically uses no correction.

What we found:
- **H3** reads 107 features — the general-purpose style reader
- **H11** dominates 66% of authors but the SAE barely decomposes it (17 features) — concentrated, opaque
- **H14** anti-correlates with dialogue/firstperson features — the formality enforcer
- **27 features don't track any head** (max |r| < 0.2) — they emerge from MLP multi-head interactions

## Step 4: Find head-independent features

Features uncorrelated with all heads are the most interesting — they represent structure that no single attention head controls. In a 1-layer model:

```
residual = embedding + Σ(head_i outputs) + MLP(embedding + Σ(head_i outputs))
```

LoRA adapters only modify attention weights. The MLP is identical across all authors. So head-independent features must emerge from how the MLP nonlinearly transforms the combination of multiple heads' outputs.

Three independent lines of evidence converge:
1. **Statistical:** these features don't correlate with any head
2. **Causal (weight steering):** modifying LoRA weights along a single direction can't move them (49% = coin flip)
3. **Causal (activation steering):** injecting the direction into the residual stream DOES move them (100% win rate for simplicity)

## Step 5: Steer and validate

Each SAE feature has a decoder column — a direction in the residual stream. Adding this direction during generation nudges the model. But does the nudge produce what the label says?

**Quantitative validation:** measure a text property across 20 seeds, count how often steering moves it in the expected direction.

What steers (75-100% win rate):
- **Simplicity** — sentence length drops (9.1 → 6.0 words)
- **Complexity** — sentence length rises (8.1 → 12.2 words)
- **Dialogue** — quote marks increase (1.3 → 3.4)
- **Questions** — question marks appear throughout
- **Verse** — line breaks mid-sentence (at low scale)

What doesn't steer on the base model:
- **Dark** (atmosphere), **cozy** (warmth), **unusual_vocab** (rare words)
- **Author-specific detectors** (e.g., character names)

**But semantic features DO steer with the matching adapter.** Cozy features on the cozy adapter amplify food descriptions. Dark features on the dark adapter amplify atmosphere. The base model doesn't have the vocabulary primed, so there's nothing to amplify — but the adapter does.

| | Base model | Matching adapter | Wrong adapter |
|---|---|---|---|
| **Structural features** | Steer | Steer | Steer |
| **Semantic features** | Don't steer | **Amplify** | Don't steer |

**The structural-semantic split:** On a 21M-parameter model, structural features (syntax, punctuation) are universally steerable. Semantic features (mood, vocabulary) are adapter-specific amplifiers. The residual stream encodes structure more controllably than meaning — unless the adapter has already primed the vocabulary.

**Feature composition:** Combining structural features creates emergent effects. Questions + dialogue + simplicity together produce a conversational-questioning voice that no single feature captures. The features are independent knobs.

**Detection ≠ steering:** A feature can be a perfect detector (fires on 0.3% of tokens, exclusively on one word) without being a reliable steering direction. Monosemantic detection does not imply monosemantic generation. Broad style directions steer; narrow token detectors don't.

## Step 6: Author reconstruction via perplexity

Can you reconstruct an author's style from feature combinations? Generate text with the base model steered by an author's elevated features, then ask the adapted model: does this look more like me?

1. For each author, find which structural features are elevated (z > 1)
2. Build a steering vector weighted by z-scores
3. Generate text with the steered base model
4. Measure perplexity under the adapted model
5. Compare against baseline and random-direction control

**Results (scale=2, 4 authors):** Poe improved (17.3 → 16.5), minimalist improved (5.1 → 4.4), poet nearly improved (beats random but not baseline), questioner degenerated. 50% improved, 75% beat random.

As predicted, reconstruction works for structurally distinct authors (minimalist, Poe's complexity) and fails for authors whose features degenerate at even low scales (questioner). Scale sensitivity is a limitation — the z-scores are already built into the vector, so high-z features get pushed too hard.

---

## What makes this approach different

| Most SAE work | This approach |
|---|---|
| Label features post-hoc from activations | Label features from designed controls (synthetics) |
| Qualitative inspection of top tokens | Three-way quantitative validation |
| No connection to causal importance | Features connected to head knockouts |
| Steering examples cherry-picked | Win rates across 20 seeds |
| No multiple testing correction | Benjamini-Hochberg (FDR=0.05) |
| Large models, impressive demos | Tiny model, honest about limitations |

The contribution isn't the model or the scale — it's the validation pipeline. Synthetics as ground truth, three-way feature labeling, head-feature correlation with statistical correction, quantitative steering validation, and the structural-semantic split as a testable finding.

---

## Limitations

- **Tiny model.** 21M parameters, 1 layer. Findings may not generalize to larger models.
- **Children's stories.** Limited vocabulary and syntax. "Style" here is simpler than in general text.
- **0.54 explained variance.** The SAE captures about half the signal. Features represent the dominant structure, not the full picture.
- **Semantic features need the adapter.** Structural features steer universally, semantic features only amplify with the matching adapter.
- **No formal causal mediation.** We show correlation (feature-head) and intervention (steering), but not full causal tracing through the computational graph.
