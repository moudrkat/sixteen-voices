# Opening the Heads: SAE Features on a Tiny Transformer

The [previous experiment](ARTICLE_SIMPLE.md) found three attention heads that carry most of the style: H11 (dominant for 66% of authors), H14 (polarizing — helps some, hurts others), H3 (consistent second). But knowing *which* heads matter doesn't tell you *what* they compute. I wanted to see inside.

---

## Looking inside the residual stream

A transformer builds up a representation at each token position — a 1024-dimensional vector called the residual stream. Each layer adds information to it: attention heads read patterns, the MLP transforms them. By the end, this vector contains everything the model knows about what token comes next.

The problem: individual dimensions don't mean anything on their own. The model uses distributed representations — concepts are spread across many dimensions, mixed together. You can't read off "this dimension encodes formality" because formality lives in a pattern across hundreds of dimensions, tangled with everything else.

A **sparse autoencoder** (SAE) [1][8] learns to decompose this mixed signal into a set of **features** — directions in the 1024-dim space where each one corresponds to a recognizable pattern. Most features are inactive for any given token; only a few "fire" at once. That sparsity forces each feature to capture something specific and distinct. (For a deeper explanation, see the [SAE explainer](SAE_EXPLAINER.md).)

I used 2048 features on a 1024-dim space — an overcomplete dictionary — with **TopK activation** [10]: at each token, only the 16 strongest features fire, the rest are zero. This gives hard sparsity (0.8% density) without tuning an L1 penalty. Of the 2048 features, 314 are alive (fire on at least some tokens); the rest are dead — the model didn't need them, which is expected for a 21M-parameter network. The alive features have a mean firing rate of 4.3%, with the most selective firing on as few as 0.3% of tokens.

---

## How do you know what a feature means?

I designed 8 synthetic styles as controls — *minimalist* (short choppy sentences), *dialogue* (all conversation), *firstperson* (first-person narration), *cozy* (warm domestic), and so on. Each isolates a single property. When a feature correlates strongly with one of these, I know what it's detecting.

For each feature, I check three things:
1. **Which tokens fire it?** (what does it literally detect?)
2. **Which synthetic styles score high or low?** (which property does it track?)
3. **Do the tokens and the synthetics tell the same story?**

When all three agree, I trust the label. This three-way validation turned out to be essential — my first attempt, labeling features from author profiles alone, produced labels like "complexity" that didn't survive testing. The author profiles were real data; my abstractions were too loose.

---

## The features

Here are the ones that passed all three checks:

**Direct speech (f1779, f627)** — f1779 fires exclusively on "I" (1.9% of tokens, highest for firstperson and dialogue), f627 on "am", "was", "'m" (2.8%, highest for dialogue). Together they detect first-person and conversational register.

**Simplicity (f665)** — fires on simple constructions, 30x higher for the minimalist synthetic than average. This is the clearest style feature in the SAE, and **no attention head controls it** (max |r| = 0.13) — it emerges from multi-head interactions through the MLP.

**Dialogue (f1777, f689)** — fires on dialogue attribution and conversation markers. Highest for the dialogue synthetic, followed by fabulist and Collodi. These features steer reliably — injecting them makes fairy tales fill with quoted speech.

**Complexity cluster (f883, f993, f60)** — fires on formal/ornate prose. Consistently highest for Lear, Baker, and Poe; near-zero for minimalist and questioner. Multiple features encoding the same axis independently. Steers reliably: injecting these directions into the minimalist adapter increases average sentence length from 8.1 to 12.2 words (100% of seeds).

Most features arrange along one dominant axis: **formal/elaborate vs simple/interactive**. 90% of the variance lives in just 9 dimensions. But the features above represent genuinely distinct directions within that space — dialogue, simplicity, and complexity are separable.

![PCA scatter](../figures/sae_style_pca.png)

*Authors in SAE feature space, colored by their dominant head from knockout experiments.*

---

## What each head computes

The [previous article](ARTICLE_SIMPLE.md) found which heads matter. The SAE tells us what they *read*:

![Feature-head bars](../figures/sae_feature_head_bars.png)

**H3 is the Swiss army knife.** 107 features — it reads the formal/simple axis, the speech features, the complexity direction. Everything interpretable goes through H3.

**H11 is the power tool.** It dominates style for 66% of authors, but the SAE can barely decompose it — only 17 features, compared to 107 for H3. It works through concentrated directions rather than a broad readable landscape.

**H2 and H15 are readable but redundant.** The overcomplete SAE reveals that H2 (86 features) and H15 (63 features) also carry interpretable structure — invisible with fewer SAE features. But knockout experiments show neither is ever the dominant head for any author. They're readable by the SAE but causally dispensable — other heads compensate when they're removed.

**H14 is the formality enforcer.** This was a mystery in the first article — why does H14 help some authors and hurt others? The SAE reveals the answer: H14 anti-correlates with first-person "I" (f1779), conversational verbs "am/was" (f627), and short-sentence markers (f1604). It correlates positively with rare vocabulary and formal punctuation patterns. Homer (+0.73) and Milton (+0.68) benefit because they're already formal. Shelley (-0.68) and Wilde (-0.34) get hurt because H14 fights their natural register.

**27 features are invisible to individual heads.** The strongest is f665 (simplicity, max |r| = 0.13 with any head). These emerge from how the MLP nonlinearly transforms the combination of multiple heads' outputs — patterns that only appear in the mix, not from any ingredient alone.

![How a 1-layer transformer computes style](../figures/sae_heads_roles.png)

*SAE features connect attention heads to specific author properties. Green axes are read by H3. The orange axis (simplicity/minimalism) doesn't track any single head — it emerges from multi-head interactions through the MLP.*

---

## Can you steer with these features?

Each feature is a direction in the residual stream. Adding it during generation — activation steering [6] — nudges the model's predictions. But does the nudge actually produce what the label says?

**Poe + simplicity** — gothic prose stripped to bare bones:

> **Baseline:** *"and the trees began to have to stop him from his bed. The dark and sky wept. The dark sky above the clouds seemed to go away"*
>
> **Steered:** *"It was dark. I went to sleep. It was dark. I woke up. It was dark. We could find a car. It was dark and it was night."*

The overwrought gothic atmosphere compressed into blunt, repetitive statements. Average sentence length drops from 23.9 to 4.9 words — a 5x compression, 20/20 seeds.

**Grimm + dialogue** — fairy-tale narration fills with conversation:

> **Baseline:** *"a little frog who was very sad. He sailed on, he had had a dream of sailing on it."*
>
> **Steered:** *"a little frog. The frog loved to bounce... the girl said, 'I have to go to the pond!' So the girl asked her father"*

Third-person narration becomes characters actually talking. Scale matters — at scale 5 this works cleanly; at scale 12 it degenerates into repetitive speech fragments.

**What breaks:** Poe + dialogue degenerates ("spirit spirit spirit"). Pushing an author too far from their register produces repetitive collapse. Steering works best as contrast — moving an author *away* from their natural voice, not amplifying it.

**What steers well and what doesn't:** Broad style directions (simplicity, complexity, dialogue) steer reliably — 75-100% win rate across 20 seeds. Narrow token-level features (e.g. character name detectors) do not steer, even when they detect perfectly. Monosemantic detection does not imply monosemantic steering.

---

## What doesn't work — and what that proves

I also tried modifying LoRA weights along feature directions — task arithmetic [7]. This doesn't work: targeted features increase only 49% of the time (coin flip).

This failure is actually informative. The features that *do* move under weight steering are all H3-correlated. The head-independent features (f665, the simplicity axis) barely budge. Since those features don't track any single head (correlation) and weight steering can't reach them (intervention), it confirms: this axis emerges from multi-head interactions that no single-direction weight modification can capture. Activation steering bypasses this by adding vectors directly to the residual stream.

---

## What we found

**The model computes style through three mechanisms:**

**H11** — the power tool. Dominant for most authors, but concentrated and hard to decompose.

**H3** — the Swiss army knife. 107 interpretable features covering the full style landscape.

**H14** — the formality enforcer. Helps formal authors, fights informal ones. Mystery from the first article: solved.

**And 27 axes that no single head controls** — simplicity, cultural register, and narrative mode, emerging from multi-head interactions through the MLP. Invisible to knockout experiments, discoverable only through the SAE.

**The features are real but shallow.** They detect word-level patterns — clause boundaries, speech markers, formality cues — not abstract "style." But those patterns track author identity well (every author has features at 8.7 standard deviations above the mean), they're validated against designed controls, and the broad style features steer reproducibly (100% win rate for simplicity and complexity, 75% for dialogue across 20 seeds each). On a 21M-parameter children's story model, that's the level of structure you get.

![Style space with steering directions](../figures/sae_style_space_arrows.png)

For methodology details, statistical choices, and the complete feature catalog, see the [technical report](TECHNICAL_REPORT_SAE.md).

---

## Try it yourself

```bash
# Train SAE with TopK sparsity
uv run python scripts/train_sae.py --activation topk --k 16 --n-features 2048 --epochs 10 --output outputs/sae_topk16_2048

# Analyze features vs heads
uv run python scripts/analyze_sae.py --sae-dir outputs/sae_topk16_2048
uv run python scripts/analyze_sae_features_v2.py --sae-dir outputs/sae_topk16_2048

# Steer from command line
uv run python scripts/steer_sae_features.py --sae-dir outputs/sae_topk16_2048 --author poe --features 665:+15
uv run python scripts/steer_sae_features.py --sae-dir outputs/sae_topk16_2048 --author grimm --features 1777:+5 689:+5

# Run all steering experiments
uv run python scripts/sweep_sae_steering_topk.py
```

Previous article: [Sixteen Voices](ARTICLE_SIMPLE.md)

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