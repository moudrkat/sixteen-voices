# Opening the Heads: SAE Features on a Tiny Transformer

The [previous experiment](ARTICLE_SIMPLE.md) found three attention heads that carry most of the style: H11 (dominant for 66% of authors), H14 (polarizing — helps some, hurts others), H3 (consistent second). But knowing *which* heads matter doesn't tell you *what* they compute. I wanted to see inside.

---

## Looking inside the residual stream

A transformer builds up a representation at each token position — a 1024-dimensional vector called the residual stream. Each layer adds information to it: attention heads read patterns, the MLP transforms them. By the end, this vector contains everything the model knows about what token comes next.

The problem: individual dimensions don't mean anything on their own. The model uses distributed representations — concepts are spread across many dimensions, mixed together. You can't read off "this dimension encodes formality" because formality lives in a pattern across hundreds of dimensions, tangled with everything else.

A **sparse autoencoder** (SAE) [1][8] learns to decompose this mixed signal into **features** — directions in the space where each one corresponds to a recognizable pattern. Most features are inactive for any given token; only a few "fire" at once. That sparsity forces each feature to capture something specific. (For a deeper explanation, see the [SAE explainer](SAE_EXPLAINER.md).)

I used **TopK activation** [10] — at each token, only the 16 strongest out of 2048 features fire, the rest are zero. My first SAE had 99% firing rate — basically not sparse at all. TopK fixed it. The sparsity matters.

---

## The pipeline

The approach has a specific order, and the order matters:

1. **Design synthetic styles** — *minimalist*, *dialogue*, *questioner*, *cozy*, and others, each isolating one property. These exist before the SAE.
2. **Train the SAE** on the base model's residual stream — running all 77 authors' texts through the unadapted model and collecting activations.
3. **Label features with synthetics** — which features correlate with which control? Cross-check with the actual tokens that fire each feature. Only label when both agree.
4. **Connect to heads** — correlate features with knockout scores from the first experiment.
5. **Steer and measure** — inject feature directions during generation, measure text properties across 20 seeds.

The synthetics are the key — they existed before the SAE, so the labels are grounded. Here's what they sound like:

> **Minimalist:** *"A cat sat. It saw a bird. The bird flew. The cat watched. Then it slept."*
>
> **Dialogue:** *""Are you going to eat me?" asked the rabbit. "I have not decided yet," said the fox. "What do you think I should do?""*
>
> **Questioner:** *"Have you ever wondered why the sky is blue? Why is it not green? Why is it not purple? Does the sky change its mind?"*
>
> **Cozy:** *"The kitchen smelled of cinnamon and warm bread and honey. Grandmother stood at the stove, stirring a big pot of soup with a wooden spoon."*

Each isolates one property. When a feature correlates with "questioner" and its top tokens are question marks, I know what it detects.

My first attempt, labeling from author profiles alone, produced labels like "complexity" that didn't survive testing. The author profiles were real data; my abstractions were too loose. The synthetics fixed that.

---

## What the SAE found

Out of 2048 features, 314 are alive — the rest are dead, which means the model only needs about 300 directions to represent style. Most features arrange along one dominant axis: formal/elaborate on one end, simple/interactive on the other. 90% of the variance lives in just 9 dimensions.

But within that space, the features split into two kinds. **Structural features** control syntax — sentence length, punctuation, line breaks, question marks. There are only about a dozen of these, but they're the ones you can steer with. **Semantic features** detect content — dialect, atmosphere, food descriptions, character voices. There are hundreds, and they're what make each author unique.

More on that split later — first, the heads.

---

## What each head computes

The [previous article](ARTICLE_SIMPLE.md) found which heads matter. The SAE tells us what they *read*:

![Feature-head bars](../figures/sae_feature_head_bars.png)

**H3 is the Swiss army knife.** Over a hundred features — it reads the formal/simple axis, speech patterns, complexity. Everything interpretable goes through H3.

**H11 is the power tool.** It dominates style for 66% of authors, but the SAE can barely decompose it. It works through concentrated directions rather than a broad readable landscape.

**H14 is the formality enforcer.** This was a mystery in the first article — why does H14 help some authors and hurt others? The SAE reveals the answer: H14 anti-correlates with first-person "I", conversational verbs, and short sentences. It pushes the model toward formal prose. Homer and Milton benefit because they're already formal. Shelley and Wilde get hurt because H14 fights their register. Mystery solved.

**27 features are invisible to individual heads.** The strongest is a simplicity direction — no attention head controls it. It emerges from how the MLP nonlinearly transforms the combination of multiple heads' outputs. Weight steering can't reach it (coin flip). Activation steering can (100% win rate). Three independent lines of evidence that this axis lives in the MLP, not in any head.

![How a 1-layer transformer computes style](../figures/sae_heads_roles.png)

---

## Two layers of style

The features answered the head question. But they also revealed something I wasn't looking for. The SAE decomposes style into two layers:

**Structural features** — simplicity, complexity, dialogue, questions, verse line breaks. A small number of directions shared across all authors. These steer reliably on any model because they map to specific tokens: periods, question marks, quotes. Inject the simplicity direction into Poe and sentence length drops from 23.9 to 4.9 words — every seed.

**Semantic features** — what makes each author unique. And the SAE finds surprisingly specific patterns:

*"not quite a smile and not quite a frown"* — dark's uncanny negation feature.
*"looking in, looking in, looking in, searching"* — dark's obsessive observation.
*"steam rose from the meat and the potatoes were crisp"* — cozy's food feature.
*"wool soft against her fingers"* — cozy's tactile comfort. A different feature from the food one.
*"purring, not growling," said Alice* — Carroll's Wonderland dialogue.

Three separate features for "cozy" alone — food, color, tactile warmth. The SAE decomposes the concept more finely than my designed label.

**Every author is primarily semantic.** Harris has zero elevated structural features and forty semantic ones. Carroll: zero structural, thirteen semantic. What makes an author unique isn't sentence length — it's content. No head specializes in one type or the other — both structural and semantic features flow through the same heads.

---

## Steering

Each feature is a direction in the residual stream. Adding it during generation nudges the model. But does the nudge actually produce what the label says?

**Poe + simplicity** — gothic prose stripped to bare bones:

> **Baseline:** *"and the trees began to have to stop him from his bed. The dark and sky wept. The dark sky above the clouds seemed to go away"*
>
> **Steered:** *"It was dark. I went to sleep. It was dark. I woke up. It was dark. We could find a car. It was dark and it was night."*

Average sentence length: 23.9 → 4.9 words. 20/20 seeds.

**Grimm + dialogue** — fairy-tale narration fills with conversation:

> **Steered:** *"a little frog. The frog loved to bounce... the girl said, 'I have to go to the pond!' So the girl asked her father"*

**Grimm + questions** — fairy tales become interrogative:

> **Steered:** *"'It is like a little frog?' 'I want to be that?' said the frog, 'I go to the mill??'"*

The SAE also learned to distinguish three types of newline — verse line breaks, paragraph breaks, and chapter headings. Same character in the text, three different features.

**What breaks:** Poe + dialogue degenerates ("spirit spirit spirit"). Steering works best as contrast — moving an author *away* from their natural voice.

**Structural features steer universally.** Simplicity and complexity: 100% win rate across 20 seeds. Dialogue: 75%. They work on any model.

**Semantic features only steer with the right adapter.** Injecting cozy features into the cozy adapter: *"She stirred and stirred and stirred, and the cat smelled the cake and the pots."* Same features on the base model: nothing. The adapter has the vocabulary primed; the features push further along that direction. The base model doesn't have "cozy" words weighted up, so there's nothing to amplify.

You can also **compose** structural features: questions + dialogue + simplicity together produce a conversational-questioning voice that no single feature captures.

---

## What doesn't work — and what that proves

Modifying LoRA weights along feature directions — task arithmetic [7] — doesn't work (coin flip). The head-independent features barely budge. This confirms: the simplicity axis emerges from multi-head interactions that no single weight modification can capture. Activation steering bypasses this by adding vectors directly to the residual stream.

---

## What we found

**Three heads, three roles.** H11 is the power tool (dominant but opaque). H3 is the Swiss army knife (readable). H14 is the formality enforcer (mystery solved).

**27 axes no head controls** — emerging from MLP multi-head mixing. The strongest style direction in the entire model is one of these.

**Style has two layers.** Shared structural axes (steerable anywhere) and unique semantic fingerprints (detectable everywhere, amplifiable only with the matching adapter). The structural-semantic split lives in the features, not in the heads.

**One more question: when you fine-tune the model for Poe, does it learn new representations or just turn up existing ones?** LoRAs amplify — they don't create. 98.8% of features in any adapted model already exist in the base model. Style is latent. Fine-tuning selects and reshapes, it doesn't construct.

![Style space with steering directions](../figures/sae_style_space_arrows.png)

For methodology details, statistical choices, and the complete feature catalog, see the [technical report](TECHNICAL_REPORT_SAE.md). For the full pipeline design, see the [methodology doc](METHODOLOGY_SAE.md).

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

# Interactive app
streamlit run demos/app_features.py
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
