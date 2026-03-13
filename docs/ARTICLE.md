# Sixteen Voices: What Happens Inside Attention Heads When You Adapt for Style

*A mechanistic interpretability experiment with LoRA-adapted transformers*

---

You know what is beautiful about tiny models? That they are tiny.

Twenty-one million parameters. One attention layer. Sixteen heads. You
could literally put this model in your pocket, if it existed physically.
And yet — no one understands what is happening inside.

So I made an experiment. Eighty-two LoRA adapters, each trained on a
different author — Carroll, Poe, Lovecraft, the Brothers Grimm,
Melville, Yeats — all squeezed through the same sixteen heads.

The question was simple: **do different authors use different heads?**

It turns out some heads matter a lot and others don't — and the pattern
is different per author. When you isolate a single head's contribution,
you can measure how much of the adaptation it carries.

A disclaimer: the model was pre-trained on TinyStories, a dataset of
short children's stories. It will never write like Poe. But it *can*
shift its output distribution in measurably different directions for
each author — and a model this small lets you draw the full map, which
you simply can't do with a hundred-layer transformer.

Here's what the adaptation looks like. Same prompt, same seed
(prompt: *"It was a dark and stormy"*, seed=42):

> **Base model:** day. But I was brave and strong." So, the little
> girl said, "I will get us. I'm strong." And everyone cheered, and
> the little girl made sure she was happy. The End.
>
> **Poe:** , and the trees began to have to stop him from his bed. The
> dark and sky wept. The dark sky above the clouds seemed to go away
> and, in the night above the clouds.
>
> **Carroll:** , and the sky was getting dark. Alice asked her, "Why,
> dark and I am dark. I get here. I am inside the clouds, and I don't
> know what is inside! I think there is a big storm coming..."
>
> **Grimm:** , and the trees began to rot. The wind stopped, and the
> leaves grew again, and the leaves were still in the wind, and the
> clouds were moving, and the sun was shining and the wind was strong.
>
> **Shelley:** y and dark a little house in the woods, but it was not
> like dark and damp. I wanted to get to sleep, but I remained in the
> darkness of the house.
>
> **Wilde:** night and the sky was grey with stars. It was a very loud
> and dark and a bright blue. The blue sky made the evening blue. The
> rain stopped, and the ground was wet and soft.
>
> **Homer:** y meowed to me. I am the sky because, of the dark and
> dependable sunset, and I am here. I am not happy because I am vain
> in the midst of vain and the other people of God as a god.
>
> **Stoker:** , and it was not nice to sleep. I did not know how to
> sleep and would sleep again. I was not in the storm, and I lay down
> in the dark, and I had a feeling of strength and a night.
>
> **Minimalist** (synthetic): night. The trees began to tremble. When the storm was
> over, the people were scared. The wind was so strong that it made
> the storm bad.
>
> **Poet** (synthetic): night and the sky was grey and the wind was
> blowing, / and the dark and the rain was deep. / The wind was strong,
> / and it looked like the night.
>
> **Dialogue** (synthetic): night." "What do you know?" asked the moon.
> "I know sky," said the storm. "Many things fly up." "The moon is not
> there. Sometimes."
>
> **Lear:** , And the Waddle!

None of this is good prose. But the distributions are measurably
different — Poe gets weeping skies, Carroll gets Alice wondering aloud,
Grimm gets nature cycles, Shelley gets first-person interiority, Homer
gets gods and vainglory, Poet gets line breaks, Dialogue gets pure
back-and-forth, and Lear gets... Lear. That's what we're working with.

---

## The Setup

### The model

[→ Figure: architecture.png]

TinyStories-1Layer-21M. Input goes through one attention layer with
16 heads (64 dimensions each), then out. Each head independently
decides what to attend to and what to output. The 16 outputs are
concatenated and projected back to produce the next token.

That's the whole model. One layer. Sixteen moving parts.

### LoRA: tiny adapters

Instead of retraining the full model, LoRA adds a small bypass to the
frozen weights: W·x + B·A·x, where A (8×1024) and B (1024×8) are the
only things we train. That's 32,768 parameters per adapter — 0.15% of
the model. We apply this to the Q and V weight matrices only.

The key insight: ΔW = B·A has the same shape as the original weight
matrix. It can be **sliced by head** — rows 0–63 belong to head 0,
rows 64–127 to head 1, and so on.

### The experiment

**82 adapters.** 69 real authors from Project Gutenberg (Poe, Carroll,
Grimm, Melville, Yeats...) and 13 synthetic controls — texts generated
with a specific style constraint rather than from a real author.
The controls include things like *minimalist* (short, blunt sentences),
*dialogue* (pure conversation), *poet* (line breaks and verse), *reporter*
(news-style prose), *repeater* (repetitive patterns), and others.
They test whether the model picks up structural patterns, not just
authorial vocabulary. All trained on CPU.

**Per-head knockout.** Keep only one head's ΔW block, zero the rest,
reconstruct valid LoRA matrices, and measure perplexity. Recovery of
1.0 = this head alone recovers the full adaptation. Negative = this
head alone makes things *worse* than no adapter at all.

Do this for all 82 authors × 16 heads and you get the main result.

---

## Results

[→ Figure: knockout_heatmap.png — 82 authors × 16 heads, clustered]
[→ Figure: knockout_strip.png — per-head recovery distribution]

The 82×16 knockout matrix shows that most heads don't matter much, but
two stand out.

**H11 is the backbone.** Mean recovery 0.338, best head for 41 of 82
authors. It probably carries basic coherence rather than anything
author-specific.

**H14 is polarizing.** The most variable head (std 0.486). For some
authors — Browne (+0.82), Melville (+0.76), Poe (+0.73) — H14 alone
recovers most of the adaptation. For others — Twain (−0.86), Burnett
(−1.39) — it makes things *worse* than no adapter at all.

**The rest cluster near zero.** H3 is a mild generalist (mean 0.30).
H6, H12, H7 contribute almost nothing.

A caveat on the numbers: recovery scores for a single author sum well
above 1.0 (median 2.17 across all 82 authors) because heads interact —
isolating one head overstates its contribution by roughly 2×. But the
inflation is fairly uniform across heads: when you normalize each
head's recovery as a fraction of the author's total, H11 still carries
23% and the bottom tier (H6, H9, H12) stays at 1–2%. The ranking
survives; just don't trust the absolute numbers.

### Is this real? Sanity check with random LoRAs

A random rank-8 matrix also puts different amounts of weight in
different head slices by chance. So "having head preferences" alone
doesn't mean anything was learned.

What random adapters don't produce is **consistency**. Across 5 random
seeds, the same author's "best head" was different every time (Shelley:
H6, H1, H11, H14, H2). Trained adapters converge — 41 of 82 agree on
H11. That convergence is learned, not a geometric artifact.

(A limitation: we verified consistency *across authors*, not across
multiple training runs of the same author. That would be a stronger
test.)

### H14: what's going on?

The H14-positive authors tend to be archaic or ornate writers. The
H14-negative authors tend to be folk tales and colloquial prose. It's
tempting to call this a "register axis" — formal vs. colloquial. But
authors whose prose is already close to TinyStories (children's
stories) need less adaptation, and H14 might just encode "distance
from pretraining distribution." The pattern is real; we don't know
what it means.

### Head transplant

[→ Figure: transplant.png]

We took Poe's H14 — his strongest head — and grafted its ΔW rows into
other authors' adapters. Most donor-host pairs produce garbage — these
three worked (same prompt and seed as the samples above).

The pattern: each host keeps its structure (Alice's dialogue, Grimm's
fairy tale rhythm, Minimalist's short sentences) but the vocabulary
shifts toward storm, thunder, darkness, weeping. The PPL cost is
modest (15–22% increase). The Minimalist transplant even breaks the
model a bit ("Ipt and weeped") — a 21M-param model being pushed
outside its comfort zone.

The shifts are consistent across all three hosts, and consistent with
what H14 does in Poe's own adapter. A single head's weight rows carry
a signal that survives transplantation.

---

## What this is and what it isn't

This is a toy experiment on one pretrained checkpoint. All findings —
H11's dominance, H14's polarization — are properties of *this model*.
A different pretraining seed would shuffle which head does what.

The clean decomposability comes from having one layer — every head
writes directly to the output, no cross-layer interaction. In a deeper
model, the same experiment would be far messier. That's a feature of
the architecture's simplicity, not a finding about attention in general.

It's a playground — small enough to see everything, useful for testing
interpretability tools before you try them on something real.

---

## What's next

- **OV circuit analysis.** We have the full W_V · W_O for each head. Instead of just observing that H14 is polarizing, we could compute *what* it promotes — which input tokens get boosted into which output tokens, and how each adapter changes that.
- **Test the pretraining distance hypothesis.** Correlate each author's token distribution divergence from TinyStories with their H14 recovery score. One scatter plot could settle whether H14 encodes "register" or just "how far from the training data."
- **Activation patching.** The current knockout operates on weights. Patching head activations during inference would give causal contributions, not weight-space proxies. Better tool, and one that actually scales to deeper models.
- **Hypernetwork for LoRA prediction.** Train a model that takes an author embedding and predicts adapter weights directly. Probably too ambitious for a 1-layer base model — but if it works, it would show whether the head structure is predictable from the text alone.
- **2-layer model.** Repeat the experiment on TinyStories-2Layers-33M (same 16 heads, same 1024 hidden dim, 2 layers). Does cross-layer composition break the clean per-head decomposability? Do the same heads specialize?

If any of this sounds interesting and you want to contribute — PRs,
ideas, or just poking holes in the methodology — you're welcome. The
code and all 82 adapters are in the repo.

---

## Figures

| Figure | Script | Description |
|--------|--------|-------------|
| Model architecture | `fig_architecture.py` | TinyStories-1Layer with 16 heads |
| Knockout heatmap | `fig_knockout_heatmap.py` | 82 × 16 recovery matrix, clustered |
| Knockout strip plot | `fig_knockout_heatmap.py` | Per-head recovery distribution |
| Head transplant | `fig_transplant.py` | Before/after text: Poe's H14 grafted into 3 hosts |

---

## References

[1] R. Eldan and Y. Li, ["TinyStories: How Small Can Language Models Be
and Still Speak Coherent English?"](https://arxiv.org/abs/2305.07759), *arXiv*, 2023.

[2] E. J. Hu et al., ["LoRA: Low-Rank Adaptation of Large Language
Models"](https://arxiv.org/abs/2106.09685), *ICLR 2022*.

[3] P. Michel, O. Levy, and G. Neubig, ["Are Sixteen Heads Really Better
than One?"](https://arxiv.org/abs/1905.10650), *NeurIPS 2019*.

[4] E. Voita et al., ["Analyzing Multi-Head Self-Attention: Specialized
Heads Do the Heavy Lifting, the Rest Can Be Pruned"](https://arxiv.org/abs/1905.09418), *ACL 2019*.

[5] N. Elhage et al., ["A Mathematical Framework for Transformer
Circuits"](https://transformer-circuits.pub/2021/framework/index.html), *Transformer Circuits Thread*, Anthropic, 2021.

[6] A. Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), *NeurIPS 2017*.