# Feature Catalog: Overcomplete TopK SAE (2048 features, k=16)

Complete reference for all characterized features. SAE trained on TinyStories-1Layer-21M residual stream, 77 authors + 8 synthetic styles.

Source data: `outputs/sae_topk16_2048/`. Scripts: `scripts/analyze_sae.py`, `scripts/classify_sae_features.py`, `scripts/sweep_sae_steering_topk.py`.

---

## Overview

| | Count |
|---|---|
| Total features | 2048 |
| Alive (firing >0.1%) | 314 |
| Dead | 1674 (82%) |
| Mean firing rate (alive) | 4.3% |
| Most selective alive feature | 0.2% (f1518, "Marilla") |
| Broadest alive feature | 37.9% (f113, mixed content words) |

### Monosemanticity breakdown ([full audit](MONOSEMANTICITY_AUDIT.md))

| Category | Count | % |
|---|---|---|
| Monosemantic (interpretable concept) | ~25 | 6% |
| Monosemantic (BPE token level) | 84 | 21% |
| Function-word detectors | 86 | 22% |
| Polysemantic | 126 | 31% |
| Near-dead | 104 | 26% |

---

## Validated features

These passed three-way validation: token-level activation + synthetic style correlation + author profile consistency.

### Structural features (steer on base model)

| Feature | Fires on | Rate | Top authors | Steers? | Evidence |
|---|---|---|---|---|---|
| **f665** | Periods in short prose | 8.9% | minimalist (11.2x), poet, simple_vocab | **Yes — 100%** win rate | Sentence length 9.1 -> 6.0 words (20 seeds). Closed-loop 80% vs 20% random. **Head-independent** (max r = 0.13). Weight steering fails (coin flip). |
| **f883 + f993 + f60** | Formatting/indentation in ornate prose | 2.2-4.4% | lear, baker, poe, gibbon | **Yes — 100%** win rate | Sentence length 8.1 -> 12.2 words on minimalist (20 seeds). Closed-loop 70% vs 20% random. Note: fires on whitespace artifacts, not "complexity" per se. |
| **f1777** | Post-quote attribution patterns | 2.7% | dialogue, fabulist, collodi | **Yes — 75%** win rate | Quote count 1.3 -> 3.4 (20 seeds). Closed-loop 90% vs 50% random. |
| **f689** | Speaker nouns after "said the" (boy, bird, whale) | 3.0% | dialogue, baum, fabulist | Yes (with f1777) | Used in combination with f1777 for dialogue steering. |
| **f1779** | "I" exclusively | 2.4% | firstperson, shelley, dialogue | **Yes on Poe** | Shifts Poe from third-person to first-person gothic. Closed-loop test misleading (20% vs 60%) — actual text shows clear "I" injection. Anti-correlates with H14 (r = -0.42). |
| **f344** | Verse line breaks | 4.6% | poet, blake | **Yes on Blake** | Turns prose into stanza structure. Closed-loop 100% vs 70% (weak by numbers but visually dramatic). |
| **f1604** | Periods at sentence boundaries | 17.8% | firstperson, dialogue, questioner | Not tested separately | Anti-correlates with H14 (r = -0.35). |
| **f1385** | "Why" in rhetorical context | 2.0% | questioner, dialogue, maeterlinck | Weak | Closed-loop 50% vs 30% random. |

### Semantic features (need matching adapter to steer)

| Feature | Fires on | Rate | Top authors | Steers? | Evidence |
|---|---|---|---|---|---|
| **f930** | Cozy tactile comfort | — | cozy | Weak on base | Closed-loop 60% vs 40% random. Needs cozy adapter. |
| **f1988** | Cozy food descriptions | — | cozy | Weak on base | Closed-loop 40% vs 30% random. Needs cozy adapter. |
| **f1224** | Dark uncanny negation | — | dark | No on base | Closed-loop 10% vs 20% random. |
| **f562** | "looked" — dark observation | 2.6% | dark, burnett, questioner | No on base | Closed-loop 20% vs 30% random. |
| **f815** | Carroll/Grahame/Maeterlinck voice | — | carroll, grahame | No on base | Closed-loop 10% vs 10%. **Head-independent**. |
| **f1117** | Grimm/Brazilian/Russian folk | — | grimm, brazilian, russian | Not tested | **Head-independent**. |
| **f372** | Russian/Burnett/Alcott domestic realism | 7.0% | russian, burnett, alcott | Not tested | **Head-independent**. Fires on periods. |

### Detection-only features (monosemantic detectors that don't steer)

| Feature | Fires on | Rate | Top authors | Why it doesn't steer |
|---|---|---|---|---|
| **f1663** | "thou" | 6.2% | harris, byron, lofting | Token not in TinyStories output distribution. Tested on base, minimalist, wilde, poe, milton, blake, byron — no archaic output at any scale. |
| **f1767** | "thee" | 3.4% | maeterlinck, milton, shelley | Same — archaic pronouns unreachable. |
| **f1621** | "thy" | 1.4% | blake, cozy, pyle | Same. |
| **f1518** | "Mar(illa)" | 0.2% | montgomery | Character name — 10% steered vs 20% baseline (30 seeds). Narrow token detectors don't generate. |
| **f329** | Questions (mixed) | — | questioner | Closed-loop 30% vs 30% — no signal. |

---

## Head correlations

Features correlated with attention head knockout scores (Pearson r, BH-corrected FDR=0.05):

| Feature | H3 | H14 | H11 | H2 | Notes |
|---|---|---|---|---|---|
| f1779 ("I") | — | **r = -0.42** | — | — | H14 suppresses first-person |
| f627 (am/was) | — | **r = -0.41** | — | — | H14 suppresses conversational verbs |
| f1604 (periods) | — | **r = -0.35** | — | — | H14 suppresses short sentences |
| f665 (simplicity) | max r = 0.13 | — | — | — | **Head-independent** — emerges from MLP |
| f815 (Carroll) | — | — | — | — | **Head-independent** |
| f1117 (folk) | — | — | — | — | **Head-independent** |
| f372 (domestic) | — | — | — | — | **Head-independent** |

H3 reads 55 features (BH-corrected). H14 reads 10 — all anti-correlated with informal/interactive markers. H11 reads only 2 despite dominating 66% of authors. 27 features are head-independent (max |r| < 0.2).

---

## New features from monosemanticity audit

Identified during systematic scan of all 314 alive features. Not yet validated with three-way method.

### Content-word detectors

| Feature | Token | Rate | Top authors | Steering result |
|---|---|---|---|---|
| f776 | "said" | 2.6% | dialogue, norse, arabian | No visible effect (scale 15, base) |
| f680 | "went" | 6.4% | repeater, dark, simple_vocab | Not tested |
| f1662 | "old" | 2.8% | japanese, russian, hawthorne | Slight register shift (scale 15, base) |
| f975 | "mrs" | 3.4% | arabian, harris, montgomery | Not tested |
| f514 | "little" | 2.1% | baum, poet, harris, cozy | Not tested |
| f646 | "don(t)" | 2.7% | alcott, twain, jacobs | No effect on base; degenerates on homer |
| f1810 | "once" | 4.3% | montgomery, dark, maya | Not tested |
| f1632 | "told" | 2.5% | japanese, italian, russian | Not tested |
| f955 | "began" | 3.7% | firstperson, carroll, alcott | Not tested |
| f401 | "such" | 4.1% | gibbon, baker, aesop | Not tested |
| f1417 | "great" | 2.0% | sewell, dequincey, byron | Not tested |
| f1460 | "ever" | 0.5% | repeater, brazilian, barrie | Not tested |
| f1646 | "come" | 1.0% | grimm, quiroga, lofting | Not tested |

### Punctuation with clear style meaning

| Feature | Token | Rate | Top authors | Interpretation |
|---|---|---|---|---|
| f1549 | "," | 8.0% | baker, tennyson, byron, milton | Formal prose comma |
| f1010 | "," | 3.7% | carlyle, greek_myth, pater, browne | Classical prose comma |
| f1000 | "." | 3.1% | lear (5.2x), baker, poe, gibbon | Period in elaborate prose (**anti-simplicity**) |
| f1881 | "." | 12.4% | carlyle, unusual_vocab, greek_myth | Period in academic/formal prose |
| f1223 | "." | 5.7% | minimalist (3.3x), simple_vocab, fabulist | Period in simple prose (second simplicity feature) |
| f9 | "?" | 2.2% | questioner (2.7x) | Pure question mark — no steering effect (scale 15, base) |
| f746 | ";" | 2.7% | byron, jacobs, melville | Semicolon — literary formality. Subtle register shift only. |
| f1116 | "," | 6.4% | rambler, alcott, sewell, indian | Conversational comma |
| f1469 | "," | 6.2% | cozy, dark, firstperson | Intimate/warm comma |

### Archaic register cluster

| Feature | Token | Rate | Top authors | Notes |
|---|---|---|---|---|
| f1663 | "thou" | 6.2% | harris, byron, lofting | Cluster fires on Blake at 14% |
| f1767 | "thee" | 3.4% | maeterlinck, milton, shelley | Perfect detectors |
| f1621 | "thy" | 1.4% | blake, cozy, pyle | Useless for steering |

---

## Steering summary

### What works (75-100% win rate)

| Direction | Features | Best on | Win rate | Metric |
|---|---|---|---|---|
| Simplicity | f665 | any (base or adapted) | 100% | Sentence length down |
| Complexity | f883 + f993 + f60 | any | 100% | Sentence length up |
| Dialogue | f1777 + f689 | base (75%), weak on adapted | 75% | Quote count up |
| First-person | f1779 | adapted (Poe) | visible in text | "I" appears throughout |
| Verse | f344 | adapted (Blake) | visible in text | Line breaks / stanza structure |

### What doesn't work

| Direction | Features | Why |
|---|---|---|
| Archaic | f1663 + f1767 + f1621 | Token not in model's output distribution |
| Character name | f1518 | Narrow token detector — 10% vs 20% baseline |
| Questions | f329 | No closed-loop signal (30% vs 30%) |
| Dark atmosphere | f1224, f562 | Semantic — needs matching adapter |
| Carroll voice | f815 | Semantic + head-independent |

### The rule

**Steering works when the feature encodes something the model can already express.** Structural features (sentence length, punctuation, line breaks) work universally because TinyStories can write short sentences and dialogue. Archaic pronouns fail because TinyStories can't produce "thou." Semantic features need the matching LoRA adapter to prime the vocabulary. This is the same principle as Anthropic's Golden Gate Bridge experiment — that feature worked because Claude can fluently generate text about the bridge.

---

## Reproducibility

```bash
# Generate full feature report (~20-30 min on CPU)
uv run python scripts/analyze_sae.py \
    --sae-dir outputs/sae_topk16_2048 \
    --top-features 400 \
    --output outputs/sae_topk16_2048/feature_report_all.txt

# Classify all features (seconds)
uv run python scripts/classify_sae_features.py

# Run all steering experiments
uv run python scripts/sweep_sae_steering_topk.py

# Generate token activation heatmaps
uv run python scripts/fig_sae_token_heatmap.py
```

---

## Files

| Path | Contents |
|---|---|
| `outputs/sae_topk16_2048/sae_weights.pt` | Trained SAE weights |
| `outputs/sae_topk16_2048/sae_config.json` | SAE config + stats |
| `outputs/sae_topk16_2048/feature_report_all.txt` | Raw per-feature activations (all 314 alive) |
| `outputs/sae_topk16_2048/feature_classification.json` | Automated classification of all features |
| `outputs/sae_topk16_2048/feature_head_analysis.json` | Feature-head Pearson correlations |
| `outputs/sae_topk16_2048/author_feature_matrix.pt` | 77 x 2048 author-feature activation matrix |
| `outputs/sae_topk16_2048/steering_sweep.json` | Original steering experiments |
| `outputs/sae_topk16_2048/steering_sweep_new.json` | New features steering (audit round) |
| `outputs/sae_topk16_2048/steering_sweep_archaic_adapted.json` | Archaic on adapted models |
| `outputs/sae_topk16_2048/steering_sweep_firstperson_verse.json` | First-person + verse experiments |
| `outputs/sae_topk16_2048/monosemanticity_report.txt` | Human-readable audit report |
| `figures/sae_token_heatmap.png` | Multi-feature token activation overview |
| `figures/sae_token_heatmap_f665.png` | Simplicity feature across authors |
| `figures/sae_token_heatmap_f1663.png` | Archaic "thou" feature across authors |
| `figures/sae_token_heatmap_f1779.png` | First-person "I" feature across authors |