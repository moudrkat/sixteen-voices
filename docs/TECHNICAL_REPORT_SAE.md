# SAE Feature Analysis: Technical Report

Technical companion to [Opening the Heads](ARTICLE_SAE.md). This document covers the full methodology, all features, statistical choices, and the weight steering diagnostic.

Setup: TinyStories-1Layer-21M [4], 77 LoRA [5] adapters (69 real authors + 8 synthetic styles), SAE on the 1024-dim residual stream. All experiments on CPU.

---

## SAE configuration: two experiments

### Experiment 1: Undercomplete ReLU SAE (v1)

| Parameter | Value |
|---|---|
| Input dim | 1024 |
| Features | 256 |
| Activation | ReLU + L1 penalty (λ = 0.001) |
| Hook point | Residual stream (input to final layernorm) |
| Decoder normalization | Unit-norm columns [1] |
| Dead features | 0 / 256 |
| Mean firing rate | 0.99 |
| Saved to | `outputs/sae/` |

**The sparsity problem:** A mean firing rate of 0.99 means each feature fires on 99% of tokens — effectively dense. The L1 penalty at λ = 0.001 was too weak to overcome the reconstruction pressure. With 256 features and 1024 input dimensions, the undercomplete bottleneck forces the model to use all features densely to minimize reconstruction loss. The SAE found real structure (author-discriminating directions, head correlations), but it was not sparse in the mechanistic interpretability sense — features were polysemantic blends, not monosemantic decompositions.

This experiment motivated the switch to TopK.

### Experiment 2: Overcomplete TopK SAE (v2)

| Parameter | Value | Rationale |
|---|---|---|
| Input dim | 1024 | Residual stream width |
| Features | 2048 | Overcomplete (2x expansion) |
| Activation | TopK, k=16 [10] | Hard sparsity without λ tuning |
| Hook point | Residual stream (input to final layernorm) |
| Decoder normalization | Unit-norm columns [1] |
| Dead features | 1674 / 2048 (82%) | Model needs ~314 directions |
| Mean firing rate | 0.043 | Genuinely sparse |
| Explained variance | 0.54 | |
| Training tokens | 256,000 | |
| Epochs | 10 | |
| Learning rate | 3e-4 | |
| Saved to | `outputs/sae_topk16_2048/` |

**Why TopK worked:** At each token, exactly 16 features fire (0.8% density). Sparsity is architectural, not a penalty to be tuned. The most selective features fire on as few as 0.3% of tokens. Dead features are expected — a 21M-parameter model doesn't have 2048 directions worth of structure. The 314 alive features represent the model's actual complexity.

**Why overcomplete now works:** The v1 overcomplete attempt failed because ReLU + L1 didn't enforce enough sparsity — the model used all features densely regardless of dictionary size. TopK eliminates this: with only 16 active per token, the model *must* choose which features matter, forcing specialization even with 2048 slots available.

### Comparison

| Metric | v1 (256, ReLU) | v2 (2048, TopK) |
|---|---|---|
| Actually sparse? | No (99% firing) | Yes (4.3% mean) |
| Features alive | 256 | 314 |
| Most selective feature | ~99% firing | 0.3% firing |
| Head-independent features found | 2 (f33, f198) | 27 |
| Heads with significant features | H3, H14, H11 | H3, H2, H15, H14, H11 |
| Explained variance | Not measured | 0.54 |
| Steering validated? | Closed-loop 83-87% | Quantitative: 75-100% win rate |

The article uses v2 results. V1 is preserved in `outputs/sae/` for reproducibility.

---

## Feature labeling methodology

Features were labeled using a three-way validation:

1. **Token-level activation**: which tokens activate the feature most strongly across the eval corpus
2. **Synthetic style correlation**: z-scores against 8 designed control styles, each isolating a single property
3. **Author profile consistency**: which real authors score high/low, and whether this matches the token/synthetic story

The synthetic styles are the key methodological tool:

| Synthetic | Isolated property | Design |
|---|---|---|
| minimalist | Short sentences | Choppy, simple, end-stopped |
| dialogue | Conversation | All speech, back-and-forth |
| firstperson | First-person narration | "I" perspective throughout |
| cozy | Warm domestic | Fireplaces, kitchens, comfort |
| unusual_vocab | Rare words | Deliberately obscure vocabulary |
| repeater | Rhythmic repetition | Repeated phrases and structures |
| questioner | Questions | Interrogative-driven text |
| reporter | Factual statements | Declarative, informational |

A feature is only labeled when all three checks agree. Features where the token pattern and synthetic correlation tell different stories are left unlabeled.

### What didn't work: jumping to abstractions

The first attempt labeled features from author profiles alone. Feature 2 (v1) is high for Homer and Poe, low for minimalist — call it "complexity." But when steered, text changed in ways that didn't match any measure of complexity. The author separation was real; the conceptual label was too loose.

Lesson: author profiles tell you *which* authors a feature separates, not *what* property it detects. You need the token-level and synthetic evidence to close the loop.

---

## Complete feature catalog (v2 SAE)

All features that passed three-way validation:

**f1779 — First-person "I".** Fires on: "I" exclusively (1.9% of tokens). Authors high: firstperson, shelley, dialogue. Low: unusual_vocab, reporter. Anti-correlates with H14 (r = -0.42).

**f627 — Conversational verbs.** Fires on: "am", "was", "'m" (2.8% of tokens). Authors high: dialogue, lang, alcott. Low: unusual_vocab, reporter. Anti-correlates with H14 (r = -0.41).

**f665 — Simplicity.** Fires on: simple constructions (mean activation 30x higher for minimalist than average). Authors high: minimalist(11.2x), poet, simple_vocab. Low: grimm, japanese, alcott. **Head-independent** (max |r| = 0.13). Steers reliably: sentence length drops from 9.1 to 6.0 words (100% win rate, 20 seeds).

**f1777 — Dialogue attribution.** Fires on: quote-after-attribution patterns ("said, 'Take'", "replied, 'Do'"). Authors high: dialogue, fabulist, collodi. Low: unusual_vocab, greek_myth. Steers at 75% win rate.

**f689 — Conversation markers.** Fires on: dialogue-associated tokens. Authors high: dialogue, baum, fabulist. Low: unusual_vocab, reporter.

**f883, f993, f60 — Complexity cluster.** Fire on: formal/ornate prose patterns. Consistently highest for lear, baker, poe; near-zero for minimalist, questioner, repeater. Multiple features encoding the same axis independently. Steers reliably: sentence length increases from 8.1 to 12.2 words on minimalist adapter (100% win rate, 20 seeds).

**f1604 — Short-sentence marker.** Fires on: periods, sentence boundaries (14.3% of tokens). Authors high: firstperson, dialogue, questioner. Anti-correlates with H14 (r = -0.35).

**f1385 — Rhetorical questions.** Fires on: "Why" in rhetorical/dialogue context. Authors high: questioner, dialogue, maeterlinck. Low: unusual_vocab, minimalist.

### Dominant axis

Most features arrange along one axis: **formal/elaborate vs simple/interactive**. 90% of the variance lives in just 9 dimensions. The features above represent genuinely distinct directions within this space.

### Author-specific features

Every author has features scoring at 8.7σ above the global mean — highly specific detectors. Examples:
- reporter: f2007, f530 (100x above mean)
- blake: f797, f116 (120x above mean)
- montgomery: f243, f1518

These are reliable detectors but do not necessarily steer. Feature f1518 (fires almost exclusively on "Marilla" at 0.3% of tokens) is a textbook monosemantic detector, but injecting its decoder direction does not reliably produce the character (10% steered vs 20% baseline across 30 seeds). Monosemantic detection does not imply monosemantic steering — broad style directions steer well, narrow token detectors do not.

---

## Dimensionality of style space

PCA on the 77 × 2048 author-feature matrix (alive features only):

| Cumulative variance | Dimensions needed |
|---|---|
| 50% | 1 |
| 80% | 3 |
| 90% | 9 |
| 95% | 19 |

More concentrated than v1 (which needed 3 dimensions for 50%). The overcomplete SAE captures the dominant axis more cleanly.

---

## Feature-head correlations

### Statistical methodology

Each of 314 alive features was correlated (Pearson r) with each of 16 heads' knockout recovery scores. Multiple testing correction via **Benjamini-Hochberg (FDR = 0.05)** and stricter thresholds (p < 0.01, |r| > 0.3) for reporting.

**Note on field norms:** The foundational SAE papers [1][2][8][9] do not use formal multiple testing correction — they rely on activation thresholds and qualitative inspection. Recent work has flagged this as a gap [11][12][13]. Our use of BH and quantitative steering validation provides evidence beyond what statistical correction alone can offer.

### Results (BH-corrected)

| Head | BH features | Loose features | Knockout: mean | Std | Dom. N | Text grounding |
|---|---|---|---|---|---|---|
| **H3** | 81 | 107 | +0.284 | 0.136 | 1 | conv_pct(-0.49***), avg_word_len(+0.42***) |
| **H2** | 59 | 86 | +0.187 | 0.106 | 0 | conv_pct(-0.32**) |
| **H15** | 35 | 63 | +0.203 | 0.120 | 1 | conv_pct(-0.33**) |
| **H14** | 12 | 47 | +0.221 | 0.292 | 18 | i_pct(-0.31**), conv_pct(-0.39***), avg_word_len(+0.38***) |
| **H11** | 1 | 21 | +0.384 | 0.167 | 51 | none |
| **H9** | 16 | 40 | +0.094 | 0.102 | 0 | none |
| **H0** | 11 | 22 | +0.117 | 0.096 | 0 | avg_word_len(+0.26*), excl_per_k(+0.30**) |
| **H8** | 0 | 30 | +0.128 | 0.152 | 2 | avg_word_len(-0.27*) |
| **H4** | 0 | 27 | +0.134 | 0.115 | 0 | none |
| **H12** | 0 | 27 | +0.080 | 0.097 | 0 | none |
| **H13** | 1 | 22 | +0.153 | 0.076 | 0 | conv_pct(-0.28*) |
| **H1** | 0 | 21 | +0.204 | 0.121 | 1 | i_pct(-0.27*), avg_word_len(+0.30**) |
| **H10** | 0 | 1 | +0.141 | 0.116 | 3 | none |
| **H7** | 0 | 7 | +0.116 | 0.095 | 0 | none |
| **H6** | 0 | 7 | +0.074 | 0.147 | 0 | none |
| **H5** | 0 | 5 | +0.114 | 0.087 | 0 | none |

### Head clusters

Feature-sharing analysis (Jaccard similarity on loose feature sets) reveals three clusters:

**Cluster 1 — Register readers (H2, H3, H14, H15).** Share 0.3–0.5 Jaccard similarity. All correlate with conversational verb density and/or average word length. They read the same axis — conversational vs elevated register — with different causal roles. H3 reads broadest (107 features), H14 acts most decisively (dominant for 18 authors, highest std at 0.292).

**Cluster 2 — Idiosyncratic style (H0, H4, H8, H9, H12).** Share 0.2–0.3 Jaccard similarity. Harris tops nearly all of them. Partially grounded in word length and exclamation marks. Seem to detect dialectal/eccentric patterns that don't map cleanly to standard text properties.

**Isolated — H11.** Jaccard 0.0 with every other head. No text-level property predicts its effect. Dominant for 51 authors. The SAE finds features it correlates with (storytelling structure, poetic register), but these overlap with nothing else in the model and can't be measured by counting words.

**Minor — H5, H6, H7, H10, H13.** Few features (0–7 loose), not dominant for meaningful author groups.

### Key observations

**H3 is the general-purpose style reader.** 81 BH-corrected features (107 at p<0.01, |r|>0.3). Grounded in conv_pct (r = -0.49, p < 0.001) and avg_word_len (r = +0.42, p < 0.001). Reads conversational verbs, first-person pronouns, sentence markers, question patterns. Shares 45 features with H14 and 53 with H15. Reads broadly but is rarely the causally dominant head (dominant for only 1 author).

**H11 is powerful and isolated.** Dominant for 66% of authors (+0.384 mean recovery) but only 1 BH-corrected feature. Its features — storytelling structure, fairy-tale patterns — share zero overlap with any other head. No text property (I-usage, conversational verbs, sentence length, word length) significantly predicts H11's effect. Whatever it reads, we cannot yet ground it.

**H14 reads narrative perspective.** Anti-correlates with:
- f1779 (first-person "I"): r = -0.42
- f627 (conversational verbs): r = -0.41
- f1604 (short sentences): r = -0.35

Positively correlates with rare vocabulary features (f2032, f1262, f1242 — highest for unusual_vocab, egyptian, browne, melville, milton).

Grounding in training text confirms the mechanism. Two text-level properties significantly predict H14 recovery:
- Conversational verb density ("am," "was," "said," "think"): r = -0.39, p < 0.001
- First-person "I" usage: r = -0.31, p = 0.007

Sentence length does **not** predict H14 (r = +0.09, p = 0.42), nor do question marks (r = -0.08, p = 0.48) or dialogue quote density (r = -0.11, p = 0.36).

Authors with low "I" usage (Homer 0.7%, Milton 0.7%, Carlyle 0.5%) and low conversational verbs score high. Authors with high "I" usage (Shelley/Frankenstein 3.6%, Stoker/Dracula 2.8%, Wells 2.2%) score low. Homer (+0.73), Milton (+0.68) benefit. Shelley (-0.68), Wilde (-0.34) get hurt.

**H2 and H15 are readable but redundant.** Part of the register-reader cluster with H3. H2 has 59 BH features and H15 has 35, but neither is ever the dominant head (H2: 0, H15: 1). They carry interpretable structure that knockout experiments don't surface — other heads compensate for their removal.

**27 features are head-independent** (max |r| < 0.2 with any head). The strongest is f665 (simplicity, max |r| = 0.13). Others include f815 (Carroll/Grahame/Maeterlinck), f1117 (Grimm/Brazilian/Russian folk), f372 (Russian/Burnett/Alcott domestic realism).

Scripts: `scripts/analyze_head_profiles.py`, `scripts/analyze_heads_deep.py`, `scripts/analyze_h14_grounding.py`
Data: `outputs/sae_topk16_2048/heads_deep_analysis.json`, `outputs/sae_topk16_2048/h14_grounding.json`

---

## MLP interaction axis

The head-independent features pose a puzzle. In a 1-layer model:

```
residual = embedding + Σ(head_i outputs) + MLP(everything)
```

LoRA adapters only modify attention weights — the MLP is identical across all 77 authors. So variation in head-independent features must ultimately originate in attention. But:

1. No single head correlates with these features (statistical evidence)
2. Weight steering along a single direction can't move them — 49%, coin flip (causal evidence)

The most likely explanation: these features emerge from how the MLP **nonlinearly transforms** the combination of multiple heads' outputs. Different LoRA adapters shift multiple heads slightly, creating different sums, which the MLP maps to different positions along the head-independent axes.

f665 (simplicity) is the strongest example: CV of 2.50 across authors (next highest MLP feature is 1.20), fires 30x more for minimalist, and steers at 100% win rate via activation steering but is unreachable via weight steering. Three independent lines of evidence converge.

---

## Steering validation

### Quantitative methodology (v2 SAE)

For each feature group, measure a text property across 20 seeds and count how often steering moves it in the expected direction:

| Feature | Metric | Baseline | Steered | Win rate |
|---|---|---|---|---|
| **Simplicity (f665)** | avg sentence length ↓ | 9.1 words | 6.0 words | **100%** (20/20) |
| **Poe + simplicity** | avg sentence length ↓ | 23.9 words | 4.9 words | **100%** (20/20) |
| **Complexity (f883+993+60)** | avg sentence length ↑ | 8.1 words | 12.2 words | **100%** (20/20) |
| **Dialogue (f1777+689)** | quote count ↑ | 1.3 | 3.4 | **75%** (15/20) |
| **Grimm + dialogue** | quote count ↑ | 0.1 | 0.3 | 20% (4/20) |

Broad style directions (simplicity, complexity) steer perfectly. Dialogue steers well on the base model (75%) but weakly on adapted models — the quote count metric may undercount dialogue that uses non-standard punctuation ('said the frog' rather than "...").

i### Closed-loop validation (v2 SAE)

Generate steered text, run it through the SAE on an *unsteered* base model, check if the targeted feature's activation increased vs baseline. Compared against random directions of the same magnitude.

| Feature | Targeted UP | Random UP | Signal? |
|---|---|---|---|
| **Simplicity (f665)** | **80%** | 20% | Yes |
| **Complexity (f883+)** | **70%** | 20% | Yes |
| **Dialogue (f1777+)** | **90%** | 50% | Yes |
| Verse (f344) | 100% | 70% | Weak |
| Rhetorical Q (f1385) | 50% | 30% | Weak |
| Cozy tactile (f930) | 60% | 40% | Weak |
| Cozy food (f1988) | 40% | 30% | Weak |
| Dialect (f61+111) | 50% | 40% | Weak |
| Questions (f329) | 30% | 30% | No |
| First person (f1779+) | 20% | 60% | No |
| Dark negation (f1224) | 10% | 20% | No |
| Dark looking (f562) | 20% | 30% | No |
| Carroll (f815) | 10% | 10% | No |

Three features pass closed-loop solidly: simplicity, complexity, dialogue (70-90% targeted vs 20-50% random). These are the same features that pass text-metric validation (75-100% win rate). Semantic features don't pass closed-loop on the base model, consistent with the finding that they need the matching adapter.

The v1 SAE reported 83-87% closed-loop for its feature groups. The v2 SAE's structural features are in the same range (70-90%), confirming consistency across SAE architectures.

### Steering examples

**Poe + simplicity (f665, scale 15):**

> **Baseline:** *"and the trees began to have to stop him from his bed. The dark and sky wept. The dark sky above the clouds seemed to go away"*
>
> **Steered:** *"It was dark. I went to sleep. It was dark. I woke up. It was dark. We could find a car. It was dark and it was night."*

**Grimm + dialogue (f1777+f689, scale 5):**

> **Baseline:** *"a little frog who was very sad. He sailed on, he had had a dream of sailing on it."*
>
> **Steered:** *"a little frog. The frog loved to bounce... the girl said, 'I have to go to the pond!' So the girl asked her father"*

**What breaks:** Poe + dialogue (scale 10) degenerates ("spirit spirit spirit"). Steering works best as contrast — pushing an author *away* from their natural register. Scale 5-10 is usually safe; above 12 risks degeneration.

---

## Addition vs clamping: two approaches to activation steering

There are two distinct approaches to injecting SAE feature directions during generation:

### Addition (Turner et al., 2023)

Take the feature's decoder column (its direction in the residual stream), scale it, and **add** it to the activation at each generation step:

```
x_steered = x + scale · W_dec[:, feature]
```

The original representation is fully preserved — steering nudges the activation in the feature's direction without destroying other information. This is the approach used throughout this report.

### Clamping (Templeton et al., 2024)

Run the activation through the SAE encoder, **force** the target feature to a high value, then reconstruct through the decoder:

```
h = encode(x)
h[feature] = clamp_value
x_steered = decode(h)
```

This is Anthropic's approach for the Golden Gate Bridge experiment [2], where a single clamped feature made Claude unable to stop talking about the Golden Gate Bridge. Clamping keeps the result "on the SAE manifold" — the output is always a valid combination of SAE features.

### Why clamping fails on our SAE

Clamping replaces the original activation with the SAE's reconstruction. This works when the SAE captures most of the input's information (Anthropic's SAE uses 130× expansion with high reconstruction fidelity). On our 2× expansion SAE (explained variance: 0.54), the reconstruction is too lossy — clamping destroys roughly half the information in the residual stream at every generation step.

We implemented clamping and tested it on the same author × feature combinations used for addition steering:

| Author | Feature | Addition result | Clamping result |
|---|---|---|---|
| Poe | simplicity (f665) | Coherent short sentences | Rapid degeneration (repetitive fragments) |
| Carroll | simplicity (f665) | Coherent simplified prose | Degeneration within ~20 tokens |
| Grimm | dialogue (f1777+689) | Added speech patterns | Incoherent output |
| Poe | first-person (f1779) | Shift from "he" to "I" | Degeneration |

Clamping degenerates because each generation step compounds the reconstruction error. The model produces a token from a degraded representation, feeds it back in, and the next step degrades it further. With addition, errors don't compound — each step starts from the model's own clean representation and adds the same fixed nudge.

**Takeaway:** Clamping requires a high-fidelity SAE. On a small SAE (2× expansion, 0.54 explained variance), addition is the only viable approach. This is a scale-dependent limitation, not a fundamental one — a higher expansion ratio with better reconstruction would likely make clamping viable.

Script: `scripts/fig_sae_showcase.py` (contains both `steer_generate()` and `clamp_generate()` implementations)

---

## Weight steering failure

### Method

Task arithmetic [7] on LoRA weight deltas:
1. Identify authors scoring high/low on a target feature
2. Average the high-scoring authors' LoRA weight deltas
3. Subtract the low-scoring authors' average
4. Add the resulting direction to any adapter's weights

### Result

Targeted features increase only 49% of the time — coin flip.

### Diagnostic

Features that *do* move under weight steering are H3-correlated. Head-independent features (f665, the simplicity axis) barely budge. This provides **causal evidence** complementing the statistical finding:

- **Statistical**: head-independent features don't correlate with any single head
- **Causal**: a single weight-steering direction can't move them
- **Steering**: activation steering reaches them (100% win rate for f665)

Three independent methods converge: these axes emerge from multi-head interactions through the MLP.

---

## Do LoRAs create new features or amplify existing ones?

The SAE was trained on the base model's residual stream. If LoRA adapters create entirely new representations, we'd expect features to appear in the adapted model that are absent in the base model. If they amplify existing structure, the same features should fire — just reweighted.

We ran the same author text through both the base model and the adapted model, compared SAE feature activations across 9 authors (mix of real and synthetic):

| Author | Features shared | New | Correlation | Pattern |
|---|---|---|---|---|
| poe | 278/278 | 0 | 0.44 | All shared, heavy reshaping |
| grimm | 261/262 | 1 | 0.96 | Gentle adjustment |
| carroll | 264/265 | 1 | 0.97 | Gentle adjustment |
| minimalist | 236/236 | 0 | 0.98 | Pure amplification |
| dialogue | 260/260 | 0 | 0.96 | Gentle adjustment |
| homer | 233/233 | 0 | 0.42 | All shared, heavy reshaping |
| wilde | 261/261 | 0 | 0.77 | Moderate reshaping |
| questioner | 257/286 | 29 | 0.88 | Only author creating features |
| poet | 240/240 | 0 | 0.96 | Gentle adjustment |

**Result: 98.8% of adapted-model features already exist in the base model. Only 1.2% are new.** LoRAs reshape the feature landscape — amplifying some directions, dampening others — but almost never create features that weren't already present.

The one exception is questioner (29 new features). The base model rarely generates questions, so question-related feature directions may not activate on base-model text. The adapter pushes the model into a region of activation space where new features become relevant.

Poe and Homer are interesting: they share all features with the base model (0 new) but have low correlation (0.44, 0.42). The adapter dramatically reweights existing features without creating new ones — a heavy reshaping of the same feature basis.

This suggests that style, at least in this model, is already latent in the base model's representations. Fine-tuning selects and amplifies, it doesn't construct.

Script: `scripts/analyze_lora_amplification.py`

---

## Summary table

| Head | Knockout role | SAE features (BH) | What it does |
|---|---|---|---|
| **H3** | Consistent second | 55 | General-purpose style reader — reads all interpretable axes |
| **H2** | Negligible in knockout | 31 | Readable but causally redundant |
| **H15** | Minor in knockout | 23 | Readable but causally redundant |
| **H14** | Polarizing (23%) | 10 | Narrative distance enforcer — anti-correlates with "I", "am/was", short sentences |
| **H11** | Dominant (66%) | 2 | Concentrated power — almost invisible to SAE despite behavioral dominance |
| **MLP** | Invisible to knockout | 27 features | Multi-head interaction axes — simplicity (f665), folk (f1117), children's lit (f815) |

---

## References

[1] T. Bricken et al., ["Towards Monosemanticity"](https://transformer-circuits.pub/2023/monosemantic-features), Anthropic, 2023.

[2] A. Templeton et al., ["Scaling Monosemanticity"](https://transformer-circuits.pub/2024/scaling-monosemanticity/), Anthropic, 2024.

[3] N. Elhage et al., ["A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html), Anthropic, 2021.

[4] R. Eldan and Y. Li, ["TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"](https://arxiv.org/abs/2305.07759), 2023.

[5] E. J. Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685), ICLR 2022.

[6] A. Turner et al., ["Activation Addition: Steering Language Models Without Optimization"](https://arxiv.org/abs/2308.10248), 2023.

[7] G. Ilharco et al., ["Editing Models with Task Arithmetic"](https://arxiv.org/abs/2212.04089), ICLR 2023.

[8] H. Cunningham et al., ["Sparse Autoencoders Find Highly Interpretable Features in Language Models"](https://arxiv.org/abs/2309.08600), ICLR 2024.

[9] S. Marks et al., ["Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models"](https://arxiv.org/abs/2403.19647), 2024.

[10] L. Gao et al., ["Scaling and Evaluating Sparse Autoencoders"](https://arxiv.org/abs/2406.04093), 2024.

[11] Y. Benjamini and Y. Hochberg, "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing", Journal of the Royal Statistical Society B, 1995.

[12] M. Meloux et al., ["The Dead Salmons of AI Interpretability"](https://arxiv.org/abs/2512.18792), 2025.

[13] B. Heap et al., ["Sparse Autoencoders Can Interpret Randomly Initialized Transformers"](https://arxiv.org/abs/2501.17727), 2025.

[14] D. Enkhbayar, ["Which Sparse Autoencoder Features Are Real? Model-X Knockoffs for False Discovery Rate Control"](https://arxiv.org/abs/2511.11711), 2025.