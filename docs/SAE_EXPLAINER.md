# What is a Sparse Autoencoder (SAE)?

A sparse autoencoder is a tool for understanding what's happening inside a neural network. It takes the tangled, high-dimensional internal representations of a model and decomposes them into individual, interpretable **features** — each one corresponding to a recognizable concept.

---

## The problem: distributed representations

A transformer builds up a representation at each token position — a vector (in our case, 1024 dimensions) called the **residual stream**. This vector contains everything the model knows about what comes next. The problem: individual dimensions don't mean anything on their own. The model uses distributed representations — concepts are spread across many dimensions, mixed together. You can't read off "dimension 47 encodes formality" because formality lives in a pattern across hundreds of dimensions, tangled with everything else.

(In larger models, this gets worse — models can encode more concepts than they have dimensions by cramming them into overlapping directions, a phenomenon called **superposition** [Elhage et al., 2022]. On our tiny model we likely don't see that, but the basic problem of entangled dimensions is the same.)

![Distributed representations](../figures/sae_explainer_superposition.png)

*Left: raw residual stream dimensions — each one mixes multiple concepts. Right: SAE features — each captures one interpretable concept, and most are off (sparse).*

---

## The solution: learn a sparse dictionary

An SAE learns to decompose the residual stream into a set of **features** that are:

1. **Interpretable** — each feature corresponds to a recognizable pattern (e.g., "direct speech", "narrative structure", "short sentences")
2. **Sparse** — for any given token, only a few features are active. Most are zero.

The combination of interpretability and sparsity is the key insight: by forcing the representation to be sparse, you force each feature to specialize. A feature that only fires occasionally has to capture something specific and meaningful — otherwise it's wasting its limited activation budget.

---

## How it works

The SAE has three components:

### 1. Encoder

A linear map followed by an activation function. Takes the 1024-dim residual stream vector and projects it into 2048 features.

```
h = activate(W_enc · x + b_enc)
```

The activation function enforces sparsity — only a few features can be active for each token. We use **TopK** [Gao et al., 2024]: keep only the 16 highest-scoring features, zero out the rest. This replaces the older ReLU + L1 penalty approach, which produced features that weren't sparse enough on our model.

### 2. Decoder

A linear map that reconstructs the original 1024-dim vector from just the active features.

```
x̂ = W_dec · h + b_dec
```

### 3. Training objective

```
Loss = MSE(x, x̂)
```

With TopK, sparsity is enforced by the activation function itself — no L1 penalty needed. The SAE must reconstruct the input well using only 16 features at a time. This tension forces each feature to capture something specific and meaningful.

![SAE architecture](../figures/sae_explainer_architecture.png)

*The full pipeline: 1024-dim input → linear encoder → TopK (keep 16, zero rest) → 2048 sparse features (most zero) → linear decoder → 1024-dim reconstruction.*

---

## What are "features"?

Each SAE feature is a **direction** in the 1024-dimensional activation space. When we say "feature 68 fires strongly on this token," we mean the residual stream vector at that position points strongly in feature 68's direction.

The decoder columns define these directions. Each column of W_dec is a 1024-dim vector — the direction in activation space that the corresponding feature represents.

![Features as directions](../figures/sae_explainer_features.png)

*Left: features are directions in activation space. Right: authors land at positions determined by which features fire for their text. Grimm and Harris are near the "narration" direction; dialogue and Carroll are near the "direct speech" direction.*

---

## Why "sparse"?

Standard autoencoders can use all hidden units for every input — they distribute information evenly. This makes them good at compression but bad at interpretation: each unit is a meaningless mix.

Sparsity changes the game. If only 16 out of 2048 features can be active for any given token, each feature must capture something genuinely distinct. You can then look at what tokens activate a feature and give it a label: "this feature detects dialogue," "this feature detects narrative scaffolding."

In practice, we check three things to label a feature:
1. **Which tokens fire it?** (what does it literally detect at the word level?)
2. **Which known styles score high or low?** (does it correlate with designed control styles?)
3. **Do the tokens and the styles tell the same story?**

When all three agree, you can trust the label.

---

## What can you do with SAE features?

### Understand the model

By correlating features with other model components (attention heads, MLP, knockout experiments), you can map out *what* the model computes and *where*. For example: "Head 3 reads the formal-vs-simple axis through 37 features" or "Head 14 anti-correlates with interactive speech features."

### Activation steering

tEach feature direction can be used as a steering vector. During text generation, you modify the residual stream at each step to push the model toward a concept. There are two approaches:

**Addition** [Turner et al., 2023]: Take the feature's decoder column (its direction in the residual stream), scale it, and **add** it to the activation. The original representation is preserved — you're nudging it, not replacing it.

```
x_steered = x + scale · W_dec[:, feature]
```

**Clamping** [Templeton et al., 2024]: Run the activation through the SAE encoder, **force** the target feature to a high value, then reconstruct through the decoder. This replaces the original activation with the SAE's reconstruction.

```
h = encode(x)
h[feature] = clamp_value
x_steered = decode(h)
```

Clamping keeps the representation "on the SAE manifold" — the result is always a valid combination of SAE features. But it destroys any information the SAE doesn't capture. This is Anthropic's approach for the Golden Gate Bridge experiment [Templeton et al., 2024], where their SAE has 130× expansion and high-fidelity reconstruction. On our 2× SAE, clamping degenerates quickly — the reconstruction is too lossy. Addition works better here because it preserves the original activation.

![Activation steering](../figures/sae_explainer_steering.png)

*Two approaches: **Addition** preserves the original activation and nudges it in the feature's direction. **Clamping** replaces the activation with an SAE reconstruction where the target feature is forced high. Clamping requires a high-fidelity SAE — on our 2× expansion it degenerates.*

For example, adding the simplicity direction to Poe turns gothic third-person narration into stripped-down short sentences. Adding the first-person direction shifts the narrator from "he" to "I."

### Discover hidden structure

Some features reveal structure that other analysis methods miss. In our experiment, we found a "structured narration" axis that no single attention head controls — it emerges from multi-head interactions through the MLP. This axis was invisible to head knockout experiments but discoverable through the SAE.

---

## Expansion ratio

Standard practice in mechanistic interpretability uses highly **overcomplete** SAEs — many more features than input dimensions (e.g., Anthropic uses 130× for Claude). The idea: the model uses superposition to pack more concepts than it has dimensions, so you need more features to unpack them all.

We use a modest **2× expansion**: 2048 features for a 1024-dim residual stream. On this small 21M-parameter model, higher expansion ratios produced features that weren't sparse enough. The key was switching from ReLU + L1 penalty to **TopK activation** [Gao et al., 2024] — forcing exactly 16 features to fire per token, regardless of expansion ratio. Our first SAE (ReLU) had 99% of features firing — not sparse at all. TopK fixed that.

---

## Key references

- Bricken et al., [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features), Anthropic, 2023 — the foundational SAE-for-interpretability paper.
- Templeton et al., [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/), Anthropic, 2024 — scaling SAEs to large models; introduces feature clamping for steering (the Golden Gate Bridge experiment).
- Cunningham et al., [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600), ICLR 2024 — systematic evaluation of SAE feature quality.
- Gao et al., [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093), 2024 — TopK activation function for SAE sparsity control.
- Turner et al., [Activation Addition](https://arxiv.org/abs/2308.10248), 2023 — steering by adding direction vectors to the residual stream.