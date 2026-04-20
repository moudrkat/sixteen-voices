# Poster FAQ

Questions that come up about the project, with honest answers. Written as a companion to the poster — if something on it raised an eyebrow, the answer is probably here.

---

## Setup / motivation questions

### "Why this model? Why not something bigger?"

That's the point. It's one layer, 16 heads, 21 million parameters. I can see *everything*. Every head, every feature, every weight. On a bigger model you'd need to sample — here I ran exhaustive experiments across all 77 adapters and all 16 heads. 1,232 knockout experiments. Every combination.

The tradeoff is real: the effects are subtle and the model only writes children's stories. But the structure I found — head specialization, two layers of style, MLP-emergent features — those are architectural properties, not dataset artifacts. If anything, finding them in a model this small suggests they exist everywhere.

### "Why LoRA specifically? What about full fine-tuning?"

LoRA gives you surgical access for free. Each adapter is a patch on the weight matrices, and because the weights are organized by head (each head is a 64-row slice), I can knock out individual heads, transplant them between authors, blend adapters by linear interpolation. Full fine-tuning changes everything at once — you can't isolate one head's contribution.

Also: 77 adapters. Full fine-tuning would mean 77 copies of the model. LoRA adapters are tiny — they all share the same base.

### "Why 77 adapters? Isn't that a lot?"

69 real authors from Project Gutenberg plus 8 synthetic controls I wrote myself. The real authors give breadth — diverse styles, different time periods, different narrative modes. The synthetics give precision — each one isolates exactly one property (short sentences, dialogue, questions, etc.). You need both. The synthetics are what make the labeling methodology work.

### "All on CPU? Really?"

Yes. The model is 21M parameters. LoRA rank 8. Each adapter trains in a few minutes. The SAE trains in under an hour. Generation takes seconds. The whole project runs on a laptop. That was part of the appeal — interpretability without infrastructure.

---

## Reading Q1 and Q2 (plain English)

### "What does 'dominant head' actually mean?"

The model has 16 attention heads running in parallel every time it generates a token. Think of them as 16 musicians in one band, all playing into the same room. The listener (the next-token predictor) hears the combined sound.

When you train a LoRA adapter to imitate Carroll, you're telling the band "play like Carroll." You might expect all 16 musicians to adjust a little. What actually happens: the gradient routes most of the Carroll-specific adjustment into **one** musician. For Carroll it's musician #11. For Poe it's #14. The other 15 still play — they keep the rhythm, grammar, coherence — but they don't carry the author-distinctive part.

"Dominant head" = the single musician who learned to play the author's distinctive part.

Silence the dominant musician → the song still happens, just generic (rabbit story instead of Alice). Silence any other musician → the song barely changes, because she wasn't playing anything author-specific anyway.

### "How did you find the dominant head for each author? (Q1)"

For each author × each head, I ran one experiment: **keep only that head's LoRA contribution, throw away the other 15.** Then measure perplexity on the author's text.

Concretely:
1. Load the author's LoRA adapter. It's a 1024×1024 delta matrix, organized as 16 stripes of 64 rows — one stripe per head.
2. For head `h`: zero out every stripe *except* `h`'s. Re-factorize back into LoRA A/B via SVD. Inject. The adapter can now only influence head `h`; the other 15 heads behave like the base model.
3. Measure perplexity on the author's held-out text. Call it `h_only_ppl`.
4. `recovery[h] = (base_ppl − h_only_ppl) / (base_ppl − full_adapter_ppl)`

Read the number: `recovery = 0.5` means "if the adapter could only touch one head, this one alone recovers half the perplexity improvement." `= 1.0` means that one head does the entire job. `= 0` means the isolated head helps nothing.

Do this for all 16 heads × all 77 authors → 1,232 experiments → the bar chart on the poster.

Result: 51 out of 77 authors have H11 as their dominant head (highest recovery). 18 have H14. The rest are scattered. For the top cases (Tennyson, Melville) one head alone recovers 70–80% of the full adapter.

**Why this is an Experiment A thing:** this is "keep only one head's LoRA." It measures how much author style is *encoded in* that head's delta. It's a weight-space experiment.

### "What is the Q2 chart actually showing?"

Q2's chart (`figures/steering_contrast.png`) is a **different** experiment from Q1's bar chart.

Here I leave the full adapter intact, then at inference time I **scale one head's total output by a factor `s`** (ranging from 0× to 2×) via a forward hook. Then I measure perplexity on the author's text as a function of `s`.

- `s = 1.0` → normal behavior (full adapter).
- `s = 0.0` → that head is silenced (its contribution zeroed).
- `s = 2.0` → that head's contribution is doubled.

The chart plots PPL vs scale for H11, H14, and H3 across Carroll and Poe.

What you see:
- **Carroll panel**: the H11 curve is a deep V — both killing (0×) and doubling (2×) hurt PPL a lot. The H14 curve is nearly flat — you can kill H14 and Carroll's PPL barely moves.
- **Poe panel**: the curves swap. H14 is the deep V; H11 is nearly flat.

That's the **mirror asymmetry**: same mechanism, same model, different author → different head dominates. It's corpus-level (averaged over the author's full eval text, not one sampled paragraph), which is why it rules out "it's just a lucky example."

**Why this is an Experiment B thing:** the full adapter is active; we're scaling an activation, not editing weights. It measures how important a head is *to generating the author's text right now*, not how much author information is *stored in* its LoRA delta.

### "Wait — so Q1 and Q2 are different experiments?"

Yes. They should agree in direction (same heads are dominant in both) and they do — but they measure different things. Q1 is about where the author style is *encoded* in the weights. Q2 is about which heads are *load-bearing* during generation. The poster uses both because together they're stronger than either alone: Q1 shows the encoding is sparse, Q2 shows the effect is causal.

---

## Methodology questions

### "How do you validate the feature labels? Aren't they subjective?"

Three-way check. First: which tokens does the feature fire on? Look at the actual text. Second: which synthetic control author does it correlate with? Third: do those two agree? If a feature fires on question marks AND correlates with the "questioner" synthetic, it's grounded. If the tokens say one thing and the correlation says another, I don't label it.

The key point: the synthetics were designed *before* the SAE existed. They're not post-hoc rationalizations — they're ground truth that was built into the experiment from the start. Most SAE work labels features after the fact. Here the labels are designed in advance.

### "25 interpretable features out of 2048 — isn't that terrible?"

Out of 2048 slots, 314 are alive (the rest are dead — the model doesn't need them). Of those 314, about 25 fire on something a human would call a concept. The rest are subword fragments, encoding artifacts, or polysemantic blends.

Is that low? Compared to Anthropic's work on Claude, yes. But Claude has hundreds of billions of parameters. TinyStories has 21 million and writes children's stories. It genuinely doesn't *have* more than about 25 distinct stylistic concepts to decompose. A bigger SAE wouldn't help because there's nothing more to find. The model is the bottleneck, not the method.

### "0.54 explained variance — the SAE only captures half the signal?"

Yes. That means 46% of the residual stream variance is noise or structure the SAE couldn't decompose. But 0.54 is enough to find the features that matter — the steering results validate this. Simplicity steers at 100% win rate, dialogue at 75%. The features the SAE *does* find are real and causal, even if it misses others.

More training tokens or a wider SAE would probably improve this. But for a first pass on a tiny model, 0.54 was enough to get interpretable, steerable features.

### "Your first SAE failed — 99% firing rate. What went wrong?"

ReLU activation with L1 penalty (lambda = 0.001). The penalty was too weak — the model used all features on every token to minimize reconstruction loss. Effectively a dense autoencoder, not sparse. It found structure (author-discriminating directions), but features were polysemantic blends.

Switching to TopK (Gao et al. 2024) fixed it completely. TopK keeps exactly the top 16 activations per token and zeros the rest. Sparsity is architectural, not a penalty to tune. Firing rate dropped from 99% to 4.3%. The most selective features fire on 0.3% of tokens.

### "What statistical corrections did you use?"

Benjamini-Hochberg (FDR = 0.05) for the feature-head correlation matrix. That's 314 features x 16 heads = ~5,000 tests. BH controls the false discovery rate. This is actually ahead of standard practice in the SAE field, which typically uses no multiple-comparison correction at all.

### "How many seeds did you test steering on?"

20 seeds per experiment. Win rate = fraction of seeds where the steered metric moved in the expected direction. Simplicity: 20/20 (100%). Complexity: 20/20 (100%). Dialogue: 15/20 (75%). These are not cherry-picked examples — they're aggregate results.

---

## Technical / mechanistic questions

### "What do you mean LoRA changes WHAT but not WHERE?"

I compared attention patterns across all 77 adapters. The patterns — which token attends to which — are nearly identical. The adaptation lives entirely in the value projections (the V matrices). LoRA changes what information each head extracts and outputs, but the routing — which positions attend to which — stays the same.

This makes sense mechanistically: in a 1-layer model, the attention pattern is determined by the query-key interaction, which operates on the input embeddings. LoRA modifies V (and Q in principle), but the dominant signal for attention routing is the base model's learned patterns.

### "What's the V-only vs Q-only finding?"

Three separate experiments, all pointing the same way.

**1. Attention-pattern comparison (77 adapters).** I ran the same prompt through every adapted model and recorded which positions attend to which. Across all 77 adapters, attention maps are nearly identical. So Q (which together with K determines attention patterns) isn't where the adaptation is happening — all 77 adapters agree on *where* to look.

**2. Functional Q-only vs V-only knockout** (`scripts/qv_decomposition.py`). For each adapter, zero only V's LoRA (keep Q's), measure which vocabulary each head promotes, compare to the full adapter. Then do the opposite (Q-only, V zeroed). **V-only overlaps with the full adapter much more than Q-only does.** V carries the vocabulary redirection.

**3. Weight-space V/Q balance correlation across 77 authors** (`outputs/h14_vq_balance.json`). For H14 specifically, I measured each adapter's V-weight fraction vs Q-weight fraction in H14's stripe and correlated with H14 recovery:
- V-fraction ↔ H14 recovery: **r = +0.58**
- Q-fraction ↔ H14 recovery: **r = −0.45**
- V−Q balance ↔ H14 recovery: **r = +0.62**

Authors whose adapters put more weight into H14's V (not Q) are exactly the authors where H14 is load-bearing.

Three methods, one conclusion: **LoRA writes into V, not Q**. The base model's attention routing is shared across all 77 adapters; only the content each head produces changes.

### "H11 is your most important head and you can't explain it?"

Correct. H11 dominates style for 66% of authors. The SAE finds 17 features that correlate with it, but those features share zero overlap with any other head, and no text-level property (word frequency, sentence length, punctuation rate) predicts H11's effect.

My honest interpretation: H11 operates through concentrated, high-dimensional directions that the SAE's 2048-feature dictionary can't fully decompose. It might need a wider SAE or a different decomposition method. Or it might compute something genuinely non-decomposable at this scale.

I'd rather report what I can't explain than make something up.

### "How does the MLP create features that no head controls?"

In a 1-layer model: `residual = embedding + attention_output + MLP(embedding + attention_output)`. The MLP is a nonlinear transformation (two linear layers with a GELU in between). It takes the *combination* of all head outputs and can create new directions that don't exist in any individual head's output.

LoRA only modifies attention weights. The MLP weights are frozen and identical across all 77 adapters. But the MLP's *input* changes because the heads' outputs change. So the MLP creates emergent features by nonlinearly mixing what the adapted heads produce. You can't find these features by knocking out heads one at a time — they emerge from the interaction.

### "What do you mean by 'weight steering' vs 'activation steering'?"

Weight steering: modify the LoRA weights in a specific direction and generate. You're changing the model's parameters permanently for that generation.

Activation steering: keep the weights unchanged, but during generation, add a direction vector to the residual stream at every token. You're injecting a signal into the model's internal state.

The MLP features can't be reached by weight steering (49% win rate = coin flip) but can be reached by activation steering (100% win rate). This is because weight steering only modifies attention, while activation steering injects directly into the residual stream — downstream of both attention and MLP.

### "What's the relationship to Anthropic's emotion paper?"

Same core idea, vastly different scale. Anthropic found 171 emotion directions in Claude using linear probes. I found ~25 style directions in TinyStories using an SAE. Both show that directions in the residual stream correspond to interpretable concepts and causally drive behavior.

The key difference: on Claude, semantic directions (emotions) steer universally. On TinyStories, only structural directions steer universally — semantic ones need a matching adapter. The model is too small to produce those tokens from the base distribution.

This gives a testable prediction: as model size increases, more semantic features should become steerable from the base model alone, without adapters. Anthropic's results on Claude already confirm this for emotions.

---

## Skeptical / critical questions

### "Isn't this just prompt engineering with extra steps?"

No. Prompt engineering changes the *input*. Steering changes the *internal computation* — same input, same weights, different behavior. The model receives the exact same tokens. The difference is a vector added to or removed from the residual stream. You can't achieve activation steering effects by changing the prompt, because the prompt operates through the model's normal processing, while steering bypasses it.

Also: prompt engineering can't knock out a head, transplant one head from Poe into a minimalist writer, or decompose the internal state into features.

### "The effects are subtle. Is this actually useful?"

For production use? Not directly — this is a toy model. For understanding transformers? Yes. The structure I found (head specialization, two-layer style decomposition, MLP-emergent features) is architectural. If it exists in a 21M model, it likely exists in larger ones — just with more features and stronger effects. Anthropic's emotion work at scale already confirms the same geometric structure.

The value is the methodology: synthetic controls as ground truth, three-way feature validation, head-feature mapping, systematic steering evaluation. That pipeline works at any scale.

### "How do you know the head specialization isn't just an artifact of LoRA?"

Two checks. First: null baseline. Untrained LoRA patches (random initialization, no training) show no head specialization. The pattern only appears after training — so it's learned, not structural.

Second: the specialization is consistent across 77 independently trained adapters. If it were a LoRA artifact, you'd expect random assignment of heads to authors. Instead you get systematic clustering — H11 for most, H14 for elevated/formal writers. That's a pattern in the data, not the method.

### "You're just looking at surface-level statistics — word frequencies, punctuation. Is that really 'style'?"

Fair criticism. At 21M parameters, the model's concept of style *is* surface-level — sentence length, punctuation patterns, function word frequency. It doesn't have deep semantic representations of irony or unreliable narration.

But that's the model's limitation, not the method's. The SAE decomposes whatever structure the model has. On a bigger model with richer representations, the same approach would find richer features. Anthropic's emotion directions are proof of that.

And even at this scale, some features go beyond pure surface statistics. The three cozy sub-features (food, color/texture, tactile warmth) distinguish semantic content, not just punctuation patterns.

### "Couldn't you get the same results with simpler methods — just measure word frequencies per author?"

You could measure word frequencies. You'd find that Poe uses more gothic vocabulary and Homer uses more formal register. But you wouldn't find:
- That H14 specifically suppresses first-person narration (not just "rare words")
- That the strongest style direction is invisible to attention heads and emerges from the MLP
- That structural features steer universally while semantic features need adapters
- That features compose — three subtle features combine into a coherent new voice

The SAE and the knockout experiments give you causal structure, not just correlations.

---

## "What's next?" questions

### "Are you going to try this on a bigger model?"

The obvious next step is a 2-layer model — does the clean head specialization survive when heads can compose across layers? After that, scaling up, but carefully. The methodology (synthetic controls, three-way validation, systematic steering) should transfer. The question is whether a bigger model has richer features that the same SAE approach can decompose.

### "Could you use this for controllable generation?"

In principle, yes — that's what steering *is*. Pick the features you want, scale them, generate. In practice, on this model the effects are subtle and sometimes degenerate. On a bigger model with stronger features, this could be a real control interface. But that's speculative — I haven't tested it.

### "What about the hypernetwork idea?"

From the first article: train a small network that takes a text sample and predicts what LoRA weights would produce that style. If it works, the 77 adapters live on a low-dimensional manifold. Haven't done this yet — it's future work.
