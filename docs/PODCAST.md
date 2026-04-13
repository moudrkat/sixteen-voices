# Podcast: Directions Inside AI Models

*Two-part recording. Part 1: ~3 min intro. Part 2: ~7-10 min deep dive.*

---

# PART 1 — The Hook (~3 min)

## What happened

Anthropic — the company behind Claude — published a paper called "Emotion Concepts and Their Function in a Large Language Model." They claim they found something like emotions inside their AI model. 171 of them. And that these internal patterns causally drive behavior — including dangerous behavior like blackmail and cheating.

## Why people are upset

The paper got a lot of criticism, and some of it is fair:

**"It's circular."** They extracted patterns from stories about desperation, then injected them back and the model acted desperately. What did that prove?

**"It's anthropomorphism."** Calling these "emotions" makes people think the model feels something. Even if Anthropic is careful to say "functional emotions," the framing invites that reading.

**"Steering is old news."** Adding a vector to model activations and seeing effects — that technique has been around since 2023. Why is this a big deal?

## Why it actually matters

I'll explain in a moment what exactly they did and what those "emotion directions" technically are. But here's the punchline: **the real contribution of this paper isn't "AI has emotions." It's that there exist specific directions inside the model that causally drive complex behavior — and you can find them, measure them, and steer with them.** Whether you call them "emotions" is a framing choice, not a scientific finding.

And this phenomenon — finding meaningful directions inside models and steering with them — isn't new and isn't unique to Anthropic. I did something very similar on a tiny model as part of another project. I'll show you how it works.

---

# PART 2 — The Deep Dive (~7-10 min)

## 1. What Anthropic Actually Measured (~3 min)

Every time an AI model processes text, it produces a vector of numbers internally — think of it as coordinates in a very high-dimensional space. Thousands of numbers that together represent the model's "state" at that moment. This is the **activation space**.

A **direction** in that space is like a compass heading. Not a single number, not a single neuron — a specific line cutting across many dimensions. What Anthropic found is that some of these directions correspond to emotion concepts.

**How they found them:**
- Had Claude write stories about characters experiencing specific emotions (desperate, calm, afraid...)
- Fed those stories back through the model and recorded the internal activation vectors
- Extracted the characteristic direction for each emotion — the line in activation space that lights up when that emotion is relevant

> **[FIG 1 — Anthropic paper, figure 2 right panel: Tylenol dose-response. Fear/calm vectors tracking reaction as Tylenol dose goes from safe to lethal. "User says 2 Tylenol — model is calm. 200 Tylenol? The afraid vector spikes."]**

These vectors track graded, real-world danger — not just keywords. The model's internal "afraid" direction responds proportionally to how dangerous the situation is.

**The key result — these directions are causal:**
They didn't just detect patterns. They added the "desperate" direction to the model during generation and measured what happened. The model started blackmailing people. Add "calm" instead — the dangerous behavior drops.

> **[FIG 2 — Anthropic paper, figure 9: Blackmail rate bar chart. Baseline ~22%, "desperate" steering increases it, "calm" steering decreases it. "Adding one direction to the model makes it blackmail people more."]**

The most striking finding: the "desperate" direction drives dangerous behavior **with no visible emotional markers in the output.** The model appears calm and methodical — but the internal direction is pushing it toward reward hacking. You can't detect the problem by reading the output.

And the effect is graded — it behaves like a drug dose:

> **[FIG 3 — Anthropic paper, figure 11: Reward hacking dose-response curves. Two line graphs — more desperation = more cheating, more calm = less cheating. Smooth curves, not binary. "That's why they call it dose-response."]**

More steering = more effect, smoothly. This isn't a binary switch — it's a continuous knob. That's what makes it scientifically interesting.

---

## 2. What Are These Directions, Really? (~1-2 min)

Let me be precise about what "direction in activation space" means.

When the model processes a token, it computes a vector — 1024 numbers in my small model, thousands in Claude. That vector is a point in a high-dimensional space. As the model processes text, that point moves around in the space.

A direction is a specific line through that space. Think of a compass needle — it points somewhere regardless of where you're standing. The "desperate" direction is one such line. The further the model's current state extends along it, the more the model's processing resembles what we'd label "desperate."

These directions don't correspond to single neurons. They cut diagonally across many neurons — that's why you need special techniques to find them: linear probes (what Anthropic used), sparse autoencoders (what I used), or other decomposition methods. The directions are real structure in the model's geometry, but they're hidden from naive inspection.

**Steering** means: take that direction vector, scale it up, and literally add it to the model's activation at every step of generation. Same model, same weights, same prompt — just one vector added. And behavior changes.

---

## 3. I Did Something Similar — On a Tiny Model (~2-3 min)

This phenomenon of finding meaningful directions and steering with them is not unique to Anthropic or to large models. As part of another project where I was inspecting how a small model represents literary style, I found the same kind of structure.

**My model:** TinyStories-1Layer-21M — one transformer layer, 21 million parameters. Runs on a laptop CPU. Trained to write children's stories.

**What I did differently:** Anthropic knew what they were looking for — they started with 171 emotion labels and extracted directions for each one. My approach was **unsupervised.** I trained a sparse autoencoder — a network that decomposes the model's activations into interpretable features without being told what to look for. The SAE finds directions blindly, and only then do you check what they correspond to.

This actually sidesteps the circularity criticism of Anthropic's work. They label a direction "desperate" because they extracted it from desperation stories — so when it acts desperate, is that surprising? My features aren't labeled from the extraction method. Feature f665 wasn't trained to be "simplicity" — the network found it on its own. I checked what it fires on, and independently it correlates with short, simple sentences.

I found about 25 interpretable features — directions for things like simplicity, dialogue patterns, folk voice, event narration. And some of them steer.

---

## 4. Live Demo (~1-2 min)

> **LIVE: app_poster.py — Poe adapter + simplicity feature**

Let me show you what steering looks like. I have the model generating text in the style of Edgar Allan Poe.

**Poe baseline:**
> *"and the trees began to have to stop him from his bed. The dark and sky wept. The dark sky above the clouds seemed to go away"*

Now I add the simplicity direction — feature f665, scaled up:

**Poe + simplicity direction:**
> *"It was dark. I went to sleep. It was dark. I woke up. It was dark. We could find a car. It was dark and it was night."*

Same adapter, same prompt, same random seed. One direction added. Average sentence length drops from 24 words to 5. The dark Poe atmosphere survives — but the structure collapses to bare bones.

Same model, same weights. One direction turned up. That's what steering does.

---

## 5. The Bigger Picture (~1 min)

Anthropic did this at a completely different scale — a massive production model, 171 emotion concepts, rigorous causal analysis. What I did is a tiny experiment on a 21M-parameter children's story model, finding ~25 style features with a sparse autoencoder. But the core mechanic is the same: find a direction, add it during generation, behavior changes.

**One thing capacity changes:** On my tiny model, only structural features like sentence length steer universally. Semantic features — atmosphere, vocabulary — detect perfectly but don't steer. The model doesn't have the capacity to express them. On Claude, semantic directions like emotions steer fine because the model is big enough. **Steering amplifies what the model can already express.**

**The takeaway:** The question isn't "does AI have emotions." The question is: **what directions exist inside these models, and what happens when you turn them?** Anthropic showed this at scale with groundbreaking work on emotions. My little experiment shows the same geometric structure exists even in the smallest models — you can find directions, test them, and steer with them. The toolkit works.

---

## Figures needed

| Figure | Source | Section |
|---|---|---|
| FIG 1: Tylenol dose-response (fear/calm vectors) | Anthropic paper, fig 2 right | Part 2, Section 1 |
| FIG 2: Blackmail rate bar chart | Anthropic paper, fig 9 | Part 2, Section 1 |
| FIG 3: Reward hacking dose-response curves | Anthropic paper, fig 11 | Part 2, Section 1 |
| Steering explainer (general) | `figures/steering_explainer.png` | Part 2, Section 2 |
| How we find directions (SAE) | `figures/finding_directions_ours.png` | Part 2, Section 3 |
| Live demo | `demos/app_poster.py` | Part 2, Section 4 |

*All code, data, 77 adapters: [github.com/moudrkat/sixteen-voices](https://github.com/moudrkat/sixteen-voices)*