# Related Work: Mechanistic Interpretability Since 2021

A reading list of papers relevant to sixteen-voices, organized by
theme. Focus is on work that appeared after Elhage et al. 2021 [5].

---

## The foundation: transformer circuits

**A Mathematical Framework for Transformer Circuits**
Elhage et al., Anthropic, 2021
[transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html)

Shows that in a 1-layer attention-only model, the output is a linear
sum of independent per-head contributions. Each head has a QK circuit
(where it looks) and an OV circuit (what it writes). This is the
theoretical backbone of our knockout and V-Q analysis — though our
model has an MLP, so the decomposition is approximate.

**In-context Learning and Induction Heads**
Olsson et al., Anthropic, 2022
[arxiv.org/abs/2209.11895](https://arxiv.org/abs/2209.11895)

Identified induction heads — a 2-layer circuit where layer-1 marks
"the token after A" and layer-2 attends to that mark to predict
"A...A → B". This is the canonical example of cross-layer composition
(V-composition). Relevant because it shows what breaks when you go
from 1 layer to 2: heads are no longer independent.

---

## Superposition and sparse autoencoders

The big shift since 2021. Individual neurons and heads are
polysemantic — they encode multiple unrelated features. SAEs
decompose activations into monosemantic features.

**Toy Models of Superposition**
Elhage et al., Anthropic, 2022
[transformer-circuits.pub](https://transformer-circuits.pub/2022/toy_model/index.html)

Showed why superposition happens: models compress more features than
they have dimensions by encoding them in nearly-orthogonal directions.
This explains why per-head analysis has limits — a head may encode
multiple features simultaneously.

**Towards Monosemanticity: Decomposing Language Models With Dictionary Learning**
Bricken, Templeton et al., Anthropic, October 2023
[transformer-circuits.pub](https://transformer-circuits.pub/2023/monosemantic-features)

Proof-of-concept: trained sparse autoencoders on a 1-layer
transformer's MLP activations and recovered interpretable,
monosemantic features. This is the paper that launched the SAE wave.

*Relevance to us:* Our model is similar in size. We analyze heads;
they analyze MLP features. A natural extension would be to train SAEs
on our model's MLP and see whether SAE features correlate with the
per-head knockout results — i.e., do the features that change most
under LoRA live in the same heads that our knockout identifies?

**Sparse Autoencoders Find Highly Interpretable Features in Language Models**
Cunningham, Ewart, Riggs, Huben, Sharkey — ICLR 2024
[arxiv.org/abs/2309.08600](https://arxiv.org/abs/2309.08600)

Independent (non-Anthropic) confirmation that SAEs work. Established
SAEs as the standard tool. Showed SAE features are more interpretable
than PCA directions.

**Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet**
Templeton et al., Anthropic, May 2024
[transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

Scaled SAEs to a production model. Found features are highly abstract
— multilingual, multimodal, and generalizing between concrete and
abstract instances. Key message: this approach works on real models,
not just toys.

**Scaling and Evaluating Sparse Autoencoders (GPT-4)**
Gao, Dupre la Tour et al., OpenAI, June 2024
[arxiv.org/abs/2406.04093](https://arxiv.org/abs/2406.04093)

OpenAI's version on GPT-4, up to 16M features. Confirmed the approach
independently. Caveat: passing activations through SAEs causes ~10x
compute-equivalent performance degradation. Full coverage may require
billions of features.

---

## Automated circuit discovery

**Towards Automated Circuit Discovery for Mechanistic Interpretability (ACDC)**
Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso — NeurIPS 2023 (Spotlight)
[arxiv.org/abs/2304.14997](https://arxiv.org/abs/2304.14997)

Automated the manual circuit-finding process. Found 5/5 component
types in the Greater-Than circuit of GPT-2 Small, selecting 68 of
32,000 edges. Later work (ACDC++, CD-T in 2024) improved efficiency.

*Relevance to us:* Our knockout sweep is essentially a brute-force
version of circuit discovery — exhaustive rather than automated, but
feasible because the model has only 16 heads. ACDC-style methods would
let us do this on larger models.

---

## Circuit tracing (current state of the art)

**Circuit Tracing: Revealing Computational Graphs in Language Models**
Anthropic, March 2025
[transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

Companion: **On the Biology of a Large Language Model**
[transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

Combines SAE features with attribution to produce full computational
graphs of model behavior on specific prompts, applied to Claude 3.5
Haiku. Discovered intermediate representations (Dallas → Texas →
Austin for multi-hop reasoning), backward planning in poem generation.
Open-sourced May 2025.

This is the culmination of the Elhage et al. program: features +
circuits + attribution in one framework on a production model.

---

## Fine-tuning and circuits

**Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis**
Zhang et al. — ICML 2025
[arxiv.org/abs/2502.11812](https://arxiv.org/abs/2502.11812)

The most directly relevant paper to our work. Found that fine-tuning
preserves circuit *nodes* but significantly rewires *edges*. Developed
"CircuitLoRA" which assigns LoRA ranks based on edge changes. Our V-Q
mechanism maps to this: V changes modify what nodes output (node
changes), Q changes rewire where heads attend (edge changes). The
finding that Q changes break in isolation is consistent with their
observation that edge rewiring is the dominant fine-tuning mechanism.

---

## Circuits across training and scale

**LLM Circuit Analyses Are Consistent Across Training and Scale**
Tigges, Hanna, Yu, Biderman — NeurIPS 2024
[arxiv.org/abs/2407.10827](https://arxiv.org/abs/2407.10827)

Tracked circuits across 300B training tokens in models from 70M to
2.8B parameters. Task-specific circuits emerge at similar token counts
across scales. The specific heads implementing a circuit may change
during training, but the underlying algorithm is stable.

*Relevance to us:* Validates that per-head functional analysis is
meaningful. Also explains why a different pretraining seed would
shuffle which head does what — the algorithm is stable, the head
assignment is not.

---

## Possible extensions for sixteen-voices

Based on this literature, analyses that could strengthen the project
without requiring a GPU:

### Feasible and high-value

1. **Train an SAE on the MLP layer.** The model is tiny — a small SAE
   (512–2048 features) should train on CPU. Then check: which SAE
   features change most when LoRA is active? Do they cluster by head?
   This would connect our head-level findings to the feature-level
   framework that the field has moved to.

2. **Attribution patching on the MLP.** Measure how much of each
   head's contribution to the final output goes through the residual
   stream directly vs through the MLP. This would quantify how
   approximate our "linear sum" assumption really is. Simple version:
   run inference with MLP frozen at base-model activations vs with MLP
   receiving adapted attention outputs. The gap = MLP contribution.

3. **Edge analysis a la ICML 2025 [10].** For each adapter, measure
   not just which heads matter (nodes) but how inter-component
   connections change (edges). In a 1-layer model the edges are:
   embedding → each head, each head → MLP, each head → logits,
   MLP → logits. Do the edge changes predict V-Q balance better
   than node-level knockout?

### Feasible but lower priority

4. **Per-head SAE feature overlap.** Train head-specific SAEs (on each
   head's output) and measure feature overlap between heads. Do H11
   and H3 (the workhorses) share features? Does H14's feature set
   change more across adapters than other heads?

5. **Logit lens through the MLP.** Project the residual stream to
   logits both before and after the MLP, for each adapter. How much
   does the MLP shift the distribution? Does the MLP shift correlate
   with knockout recovery?

### Probably not worth it

- Full ACDC circuit discovery — overkill for a 1-layer model where
  you can already enumerate everything.
- Training SAEs on attention patterns (QK space) — the monosemanticity
  work focuses on residual stream / MLP; attention-pattern SAEs are
  less established.
- Scaling to a 2-layer model — interesting but a different project,
  not a quick extension.