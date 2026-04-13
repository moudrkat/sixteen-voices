# Podcast: Directions Inside AI Models

A podcast about what "directions inside AI models" means, why Anthropic's emotion paper matters, and how we found the same structure in a tiny model on a laptop.

---

## What Anthropic found

Anthropic published a paper called "Emotion Concepts and Their Function in a Large Language Model." They found 171 internal directions inside Claude that correspond to emotion concepts — and these directions causally drive behavior.

Every time a model processes text, it builds an internal state — a vector of thousands of numbers. A **direction** in that space is like a compass heading cutting across many dimensions. Anthropic found that some of these directions correspond to emotions: desperation, calm, fear.

These vectors track graded, real-world danger — not just keywords. The model's internal "afraid" direction responds proportionally to how dangerous a situation is. User says 2 Tylenol — the model is calm. 200 Tylenol? The afraid vector spikes.

**The key result — these directions are causal.** They added the "desperate" direction to the model during generation. The model started blackmailing people. Add "calm" instead — the dangerous behavior drops. The most striking part: the "desperate" direction drives dangerous behavior with no visible emotional markers in the output. The model appears calm and methodical — but internally it's being pushed toward reward hacking.

The effect is graded — more steering, more effect, smoothly. Not a binary switch — a continuous knob.

The paper got criticism ("it's circular," "it's anthropomorphism," "steering is old news"), and some of it is fair. But the real contribution isn't "AI has emotions." It's that **specific directions inside the model causally drive complex behavior — and you can find them, measure them, and steer with them.**

---

## What are these directions?

When a model processes a token, it computes a vector — 1024 numbers in a small model, thousands in Claude. That's a point in a high-dimensional space. A direction is a specific line through that space — like a compass needle that points somewhere regardless of where you're standing.

These directions don't correspond to single neurons. They cut diagonally across many neurons — that's why you need special techniques to find them: linear probes (what Anthropic used), sparse autoencoders (what I used), or other decomposition methods. The directions are real structure in the model's geometry, but they're hidden from naive inspection.

**Steering** means: take a direction vector, scale it up, and add it to the model's activations during generation. Same model, same weights, same prompt — one vector added. Behavior changes.

---

## A tiny experiment with the same idea

As part of the [Sixteen Voices](https://github.com/moudrkat/sixteen-voices) project, I explored the same phenomenon on a much smaller scale. The model: TinyStories-1Layer-21M — one transformer layer, 21 million parameters, trained on children's stories. Runs on a laptop CPU.

Anthropic knew what they were looking for — they started with 171 emotion labels and extracted directions for each one. My approach was **unsupervised.** I trained a sparse autoencoder that decomposes the model's activations into interpretable features without being told what to look for. Feature f665 wasn't trained to be "simplicity" — the SAE found it on its own. I checked what it fires on, and independently it correlates with short, simple sentences.

This actually sidesteps the circularity criticism: my features aren't labeled from the extraction method.

I found about 25 interpretable features — directions for things like simplicity, dialogue patterns, folk voice, event narration. Some of them steer:

**Poe baseline:**
> *"and the trees began to have to stop him from his bed. The dark and sky wept. The dark sky above the clouds seemed to go away"*

**Poe + simplicity direction:**
> *"It was dark. I went to sleep. It was dark. I woke up. It was dark. We could find a car. It was dark and it was night."*

Same adapter, same prompt, same random seed. One direction added. Sentence length drops from 24 words to 5. The dark Poe atmosphere survives — but the structure collapses to bare bones.

---

## What's different at small scale

Anthropic did this at a completely different scale — a massive production model, 171 emotion concepts, rigorous causal analysis. My experiment is tiny by comparison. But one interesting thing came out of the scale difference:

On a small model, only structural features like sentence length steer universally. Semantic features — atmosphere, vocabulary — detect perfectly but don't steer. The model doesn't have the capacity to express them. On Claude, semantic directions like emotions steer fine because the model is big enough. **Steering amplifies what the model can already express.**

The takeaway: the question isn't "does AI have emotions." The question is **what directions exist inside these models, and what happens when you turn them?** Anthropic showed this at scale with groundbreaking work. This little experiment shows the same geometric structure exists even in the smallest models.

---

## Read more

**[Article 1: Sixteen Voices](https://www.linkedin.com/pulse/sixteen-voices-interpretability-experiment-tiny-kate%C5%99ina-fajmanov%C3%A1-jmfnf)** — The first experiment. I trained 77 LoRA adapters on the same tiny model (one per author style) and ran 1,232 head knockout experiments to find out which of the 16 attention heads carry style. Three heads matter, the rest don't — H11 dominates for most authors, H14 leads for a cluster of elevated writers (Poe, Homer, Milton), and H3 is a consistent second. I also tried steering heads at inference time, transplanting single heads between authors, and blending adapters to mix styles.

**[Article 2: Experiment in a Pocket](https://www.linkedin.com/pulse/experiment-pocket-opening-tiny-model-finding-knobs-kate%C5%99ina-fajmanov%C3%A1-crodf)** — The follow-up. Instead of asking "which heads matter," I trained a sparse autoencoder on the residual stream to find out what the model actually represents internally. Found ~25 interpretable features — simplicity, dialogue, first-person narration, questions. Some steer (simplicity works on every author), some only detect (archaic pronouns fire perfectly on Blake and Milton but produce nothing when injected). The key finding: the strongest style direction is invisible to all attention heads — it emerges from the MLP. This is the article that connects directly to the podcast topic.

- [Interactive demo](https://sixteen-voices.streamlit.app) — try steering the model yourself
- [GitHub repo](https://github.com/moudrkat/sixteen-voices) — all code, 77 adapters, full technical reports