# Sixteen Voices — a poster walkthrough

A narrative tour of the poster: a one-layer transformer, 77 LoRA adapters, and what it took to understand how writing style lives inside a tiny language model.

---

## The setup

The smallest language model that can still write coherent text — 21 million parameters, one transformer layer, trained on children's stories. I taught it 77 different writing styles by training lightweight patches called LoRAs — one per author. Poe, Carroll, Grimm, Homer, plus synthetic styles I designed myself. Then I spent several months taking it apart to understand how it represents style internally.

Everything ran on a laptop CPU. No GPU, no cluster, no budget.

The architecture diagram in the center of the poster is the entire model: tokens come in, get embedded into vectors, pass through 16 attention heads in parallel, then through an MLP, and out comes a prediction. One layer. Small enough that every piece can be examined.

---

## Chapter 1: The behavioral experiments

The first question was whether all 16 heads matter equally. To find out, I knocked them out — one at a time, across all 77 adapters. 1,232 experiments total.

Three heads do most of the work. **H11** is the workhorse — the dominant head for 66% of authors. **H14** leads for a smaller cluster: Homer, Poe, Milton, Lovecraft — the "elevated" writers. H11 and H14 are anticorrelated: when one matters, the other doesn't. **H3** is a consistent second for almost everyone.

Control check: untrained LoRA patches — random weights, no training — show no head specialization. So the pattern is learned, not baked into the architecture.

Scaling heads up and down at inference time gives a behavioral signature. Kill Carroll's dominant head H11 and you get a generic rabbit story. Kill Poe's dominant head H14 and you get nonsense. Style lives in specific heads.

A surprising finding: LoRA changes *what* heads output, not *where* they look. Attention patterns across all 77 adapters are nearly identical. The adaptation is entirely in the value projections — the content, not the routing.

Transplanting single heads between authors works too. Take a minimalist writer, swap in just Poe's H14 — 6% of the LoRA weights — and short sentences remain while dark vocabulary floods in. The structure stays, the content shifts.

Blending adapters by linear interpolation is mixed. Some pairs blend smoothly — Carroll into Poet, prose restructuring itself as dialogue fades and line breaks appear. Others break: Poe into Carroll produces gibberish at the midpoint. LoRA weight space isn't a smooth style space.

---

## Chapter 2: Looking inside

Knowing *which* heads matter doesn't tell you *what* they compute. H14 being important for Poe doesn't explain why.

So I trained a sparse autoencoder on the model's residual stream — the internal state after all the heads and the MLP have had their say. The SAE decomposes that 1024-dimensional vector into individual features: directions in the space, each corresponding to something specific.

Out of 2048 possible features, 314 are alive. Of those, about 25 fire on something a human would recognize. That sounds low, but a 21-million-parameter children's story model genuinely doesn't have more than about 25 stylistic concepts to decompose.

The key methodological move: synthetic control authors were designed *before* training the SAE. Minimalist — short sentences. Dialogue — all conversation. Questioner — all questions. These existed as ground truth before anyone looked inside. When a feature correlates with the questioner synthetic AND its top-activating tokens are question marks, the label is grounded — not a post-hoc guess.

The first SAE attempt was a disaster — 99% firing rate, every feature firing on every token. Not sparse at all. Switching to TopK activation fixed it: exactly 16 features fire per token, the rest are zero. Firing rate dropped to 4.3%.

---

## Chapter 3: What the heads actually do

With the SAE in place, features can be connected back to the heads from the first experiment.

**H3 is the swiss army knife.** 107 features. It reads everything — every interpretable axis I found.

**H11 is the opaque powerhouse.** It dominates style for most authors, but the SAE can barely read it — only 17 features, none overlapping with any other head. Whatever H11 computes, it resists decomposition in human terms. That's the honest answer.

**H14 is where it gets interesting.** H14 suppresses first-person narration. It anti-correlates with "I", with conversational verbs, with short sentences. It amplifies rare vocabulary. Authors who narrate from the outside — Homer, Milton, Melville — benefit from H14. Authors who narrate from the inside — Shelley's first-person Frankenstein, Stoker's diary-entry Dracula — get hurt by it. H14 is a formality enforcer. That explains the anticorrelation with H11.

And then there's the MLP. 27 features don't belong to any head. The strongest one is a simplicity direction — and no attention head controls it. It emerges from how the MLP nonlinearly transforms the combination of multiple heads' outputs. Knockout experiments can't find it. Activation steering reaches it every time — 100% win rate across 20 seeds.

---

## Chapter 4: The main finding — two layers of style

The SAE revealed that style has two layers.

**Structural features** are shared axes — universal knobs. Simplicity, complexity, dialogue, questions, verse line breaks. They work on any adapter because they map to specific tokens: periods, question marks, quotes. Steering Poe with the simplicity direction collapses his gothic prose to bare bones: "It was dark. I went to sleep. It was dark. I woke up." Sentence length drops from 24 words to 5. 20 out of 20 seeds.

**Semantic features** are what makes each author unique. Dark atmosphere, cozy food descriptions, dialect spelling, Wonderland dialogue. The SAE detects them perfectly — every time. But they only steer when the matching adapter is loaded. The base model doesn't have enough probability mass on those tokens to amplify.

The SAE even decomposes finer than designed. One "cozy" synthetic went in; three separate cozy features came out — food descriptions, color and texture, tactile warmth. Sub-structure that wasn't asked for.

Features compose. Take the base model — no adapter — and inject three features at once: questions, dialogue, simplicity. Individually they're subtle. Combined, a new coherent voice emerges: "She asked her mommy, 'Can I go outside and play?' Her mommy said, 'Yes, but you must stay on the swing.'"

---

## Chapter 5: What breaks

Not everything works. Poe + dialogue steering degenerates into "spirit spirit spirit." Archaic-pronoun features detect "thou" and "thee" perfectly in Blake and Milton, but injecting them never produces a single archaic word — the model collapses first. Perfect detectors, useless steering vectors.

Compare with Anthropic's Golden Gate Bridge experiment: clamping one feature in Claude and the model couldn't stop talking about the bridge. Claude has billions of parameters and the bridge is deep in its training distribution. TinyStories has 21 million parameters. **Steering amplifies what the model can already express.** If the base model can't produce "thou," no amount of steering will get you there.

This gives a testable prediction: on a bigger model, semantic features should steer universally without needing per-style adapters. Anthropic's emotion paper from 2025 already shows this happening — steering Claude with emotion directions works because Claude is big enough.

---

## The one-liner

A tiny transformer represents style in two layers — shared structural knobs that turn on any model, and unique semantic fingerprints that can be read everywhere but only controlled through the right adapter.