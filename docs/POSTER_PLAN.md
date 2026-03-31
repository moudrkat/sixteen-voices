# Poster: ML Prague

## Sixteen Voices: How a Tiny Transformer Represents Writing Style

*Mechanistic interpretability on TinyStories-1Layer-21M. All experiments on a laptop CPU.*

---

## The story

I took a tiny model — 21 million parameters, one attention layer, trained on children's stories — and taught it 77 different writing styles on my laptop CPU. I trained 69 real authors and designed 8 synthetic controls that each isolate one property. Then I looked inside. I found which heads matter, looked deeper with a sparse autoencoder, discovered style has two layers — shared structure you can steer and unique semantics you can detect — and proved LoRAs amplify what's already there.

---

## 1. A tiny model, 77 styles, a laptop

TinyStories-1Layer-21M. 21M parameters. 1 attention layer with 16 heads. 77 LoRA adapters — lightweight patches on the same base model. 69 real authors (Poe, Carroll, Grimm, Homer...) plus 8 synthetic controls, each isolating one property: *minimalist* (short sentences), *dialogue* (all conversation), *questioner* (all questions), *cozy* (warm domestic)...

Same prompt, different adapters:

> **Poe:** *"and the trees began to have to stop him from his bed. The dark and sky wept."*
>
> **Dialogue:** *""What do you know?" asked the moon. "I know sky," said the storm."*
>
> **Lear:** *", And the Waddle!"*

The synthetic controls matter. They are the ground truth for everything that follows.

**Figure:** text examples from adapters

---

## 2. Which heads matter

Knocked out each of the 16 heads, one at a time, measured which ones the model needs.

- **H11** — dominant for 66% of authors. The workhorse.
- **H3** — consistent second. Reads broad style structure.
- **H14** — polarizing. Helps Homer and Milton. Hurts Shelley and Wilde. WHY?

Knowing *which* heads matter doesn't tell you *what* they compute.

**Figure:** `knockout_heatmap.png`

---

## 3. Looking inside with a sparse autoencoder

The model's internal state is a 1024-dimensional vector. A sparse autoencoder decomposes it into interpretable directions. TopK activation — only 16 out of 2048 features fire per token.

The pipeline:
1. Design synthetic controls (before the SAE)
2. Train SAE on base model's residual stream
3. Label features with synthetics — three-way check
4. Connect features to head knockouts (BH-corrected)
5. Steer and measure (win rates across 20 seeds)

Synthetics are ground truth that existed before the decomposition. Labels are grounded, not post-hoc guesses.

*(First SAE had 99% firing rate — not sparse. TopK fixed it to 4.3%. The sparsity matters.)*

**Figure:** `poster_pipeline.png`

---

## 4. What each head computes

**H3 = Swiss army knife.** 107 features — reads all interpretable axes.

**H11 = power tool.** Dominant but opaque. Only 17 readable features.

**H14 = formality enforcer.** Anti-correlates with "I", "am/was", short sentences. Helps Homer (+0.73), Milton (+0.68). Hurts Shelley (−0.68), Wilde (−0.34). **Mystery solved.**

**MLP = multi-head mix.** 27 features no head controls, including the STRONGEST style direction (simplicity, f665, max |r| = 0.13 with any head). Weight steering can't reach it (49%). Activation steering can (100%).

**No head specializes in structure vs semantics.** Both types flow through the same heads. H3 carries 103 semantic and 4 structural features. The split is in the features, not the heads.

**Figure:** `poster_head_roles.png`

---

## 5. Two layers of style — the main finding

### STRUCTURAL — shared axes, universal knobs

| Feature | What it controls | Win rate |
|---------|-----------------|----------|
| Simplicity (f665) | sentence length 9.1 → 6.0 words | 100% |
| Complexity (f883+) | sentence length 8.1 → 12.2 words | 100% |
| Dialogue (f1777+) | quote marks 1.3 → 3.4 | 75% |
| Questions (f329) | declarative → interrogative | works |
| Verse (f344) | prose → verse line breaks | works |

Works on any model because they map to specific tokens: periods, question marks, quotes, line breaks.

### SEMANTIC — unique identity, adapter-specific

| Feature | What it detects | Tokens |
|---------|----------------|--------|
| Dark f1224 | uncanny negation | *"not quite a smile, not quite a frown"* |
| Dark f562 | obsessive observation | *"looking in, looking in, looking in"* |
| Cozy f1988 | food descriptions | *"steam rose from the meat"* |
| Cozy f930 | tactile comfort | *"wool soft against her fingers"* |
| Carroll f815 | Wonderland dialogue | *"purring, not growling," said Alice* |
| Harris f61 | dialect spelling | *"uz so wet dey don't"* |

Detect perfectly. Steer only with matching adapter.

### The split

Every author is primarily semantic. Harris: 0 structural, 40 semantic features. Carroll: 0 structural, 13 semantic. What makes an author unique isn't sentence length — it's content.

~14 structural features total across all heads. ~300+ semantic. Both flow through the same heads — no head specializes. The structural features are rare but universally steerable. The semantic ones are the vast majority but need the adapter.

**Figure:** `poster_two_layers.png`

---

## 6. Steering works

**Poe + simplicity** — 23.9 → 4.9 words per sentence, 20/20 seeds:

> *"The dark sky above the clouds seemed to go away..."*
> → *"It was dark. I went to sleep. It was dark. I woke up."*

**Grimm + questions** — fairy tales become interrogative:

> → *"'I want to be that?' said the frog, 'I go to the mill??'"*

**Cozy + cozy features** — semantic amplification with matching adapter:

> → *"She stirred and stirred, and the cat smelled the cake and the pots."*

**Three types of newline** — verse breaks (3%), paragraph breaks (12%), chapter headings (8%). Same character, three features.

**Figure:** `poster_steering_examples.png`

---

## 7. The bigger picture

**LoRAs amplify, they don't create.** 98.8% of features already exist in the base model. Style is latent. Fine-tuning selects, it doesn't construct.

**The strongest style direction is invisible to heads.** Simplicity emerges from MLP multi-head mixing. Activation steering reaches it. Weight steering can't. The SAE finds structure that knockouts miss.

**Structure steers universally, semantics needs the adapter.** On a 21M model, syntax is controllable, meaning is readable but not steerable from the base model. Bigger models should steer semantics too — a testable prediction.

**Figure:** `sae_style_space_arrows.png`

---

## One sentence

A tiny transformer represents writing style in two layers — shared structural axes that are universally steerable, and unique semantic fingerprints that are detectable everywhere but controllable only through author-specific adapters.

---

## Figures

| Panel | Figure | Have it? |
|---|---|---|
| 1. Adapters | text examples or `adapter_pca.png` | Yes |
| 2. Knockouts | `knockout_heatmap.png` | Yes |
| 3. Pipeline | `poster_pipeline.png` | Yes |
| 4. Head roles | `poster_head_roles.png` | Yes |
| 5. Two layers | `poster_two_layers.png` | Yes |
| 6. Steering | `poster_steering_examples.png` | Yes |
| 7. Style space | `sae_style_space_arrows.png` | Yes |

## Design notes

- Color-code: blue=H11, green=H3, red=H14, orange=MLP
- Text examples are the hook — people read stories
- Two-layer finding = centerpiece, largest panel
- "Laptop CPU" in subtitle — shows accessibility
- Readable from 2 meters — big text, big figures
- QR code to repo + interactive demo
