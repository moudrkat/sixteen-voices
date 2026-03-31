# Sixteen Voices: What a Tiny Transformer Knows About Style

Summary connecting both articles. TinyStories-1Layer-21M, 77 LoRA adapters (69 real authors + 8 synthetic controls), sparse autoencoder on the residual stream.

---

## The question

Can you take a 21-million-parameter children's story model, teach it 77 different writing styles, and then figure out *how* it represents them?

## What we did

**Article 1 — Which components matter:**
Knocked out each of the 16 attention heads, one at a time, and measured which ones hurt which authors. Found three that carry most of the style: H11 (dominant for 66% of authors), H3 (consistent second), H14 (polarizing — helps some authors, hurts others). But *what* they compute was unknown.

**Article 2 — What they compute:**
Trained a sparse autoencoder (TopK, 2048 features, k=16) on the residual stream to decompose the model's internal state into interpretable directions. Connected these features to the heads from Article 1, then steered with them.

## What we found

### Three heads, three roles

- **H11** — the workhorse. Dominant for most authors but the SAE can barely read it (17 features). It works through concentrated directions.
- **H3** — the reader. 107 interpretable features covering the full style landscape.
- **H14** — the formality enforcer. Anti-correlates with "I", conversational verbs, short sentences. Helps Homer (+0.73), Milton (+0.68). Hurts Shelley (-0.68), Wilde (-0.34). Mystery from Article 1: solved.

### The MLP axis

27 features don't belong to any attention head. The strongest — a simplicity direction — has max correlation of 0.13 with any head. Three independent methods confirm it's real: no head correlates with it (statistical), weight steering can't reach it (causal), activation steering reaches it at 100% win rate (intervention).

### Two layers of style

The SAE decomposes style into:

**Structural features** — shared axes that every author sits on. Simplicity, complexity, dialogue, questions, verse line breaks. These steer universally on any model (75-100% win rate across 20 seeds) because they map to specific tokens: periods, question marks, quotes, line breaks.

**Semantic features** — what makes each author unique. Harris's dialect ("uz so wet dey don't"), Poe's uncanny negation ("not quite a smile and not quite a frown"), three separate features for "cozy" (food, color/texture, tactile warmth), Carroll's Wonderland dialogue. Every author is primarily semantic: Harris has 0 structural and 40 semantic elevated features. These detect everywhere but only steer when paired with the matching LoRA adapter.

### LoRAs amplify, they don't create

98.8% of features in any adapted model already exist in the base model. Fine-tuning reshapes the feature landscape — amplifying some directions, dampening others — but almost never creates features that weren't already present. Style is latent in the base model's representations.

## The pipeline

The approach has a specific order:

1. **Design synthetic controls** before looking inside — minimalist, dialogue, questioner, etc.
2. **Train the SAE** — decompose the residual stream into sparse features
3. **Label features with controls** — three-way check: tokens, synthetics, cross-agreement
4. **Connect to heads** — correlate features with knockout scores (Benjamini-Hochberg corrected)
5. **Steer and measure** — quantitative validation across 20 seeds

The synthetics are ground truth that existed before the decomposition. Most SAE work labels post-hoc. Here the labels are grounded.

## Best demonstrations

**Poe + simplicity** — gothic prose stripped to bare bones:
> *"It was dark. I went to sleep. It was dark. I woke up. It was dark."*
> Sentence length: 23.9 → 4.9 words. 20/20 seeds.

**Grimm + questions** — fairy tales become interrogative:
> *"'It is like a little frog?' 'I want to be that?' said the frog, 'I go to the mill??'"*

**Cozy adapter + cozy features** — semantic amplification:
> *"She stirred and stirred and stirred, and the cat smelled the cake and the pots."*

**Three types of newline** — the SAE distinguishes verse line breaks (f344, 3%) from paragraph breaks (f1524, 12%) from chapter headings (f793, 8%).

**Three types of cozy** — food descriptions (f1988), color/texture (f29), tactile comfort (f930).

## The honest limitations

- 21M parameters, 1 layer, children's stories. "Style" here is simpler than in general text.
- 0.54 explained variance — the SAE captures about half the signal.
- Semantic features detect but don't steer on the base model — you need the adapter.
- No formal causal mediation through the computational graph.
- The features are word-level patterns, not abstract style concepts.

## One sentence

A tiny transformer represents writing style in two layers — shared structural axes (sentence length, punctuation, line breaks) that are universally steerable, and unique semantic fingerprints (dialect, vocabulary, tone) that are detectable everywhere but controllable only through author-specific adapters — and the strongest style direction in the model is invisible to attention head analysis, emerging instead from nonlinear MLP interactions.
