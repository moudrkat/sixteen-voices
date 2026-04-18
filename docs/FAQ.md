# FAQ — Sixteen Voices

Answers to questions that come up in poster discussions, code reviews, and my own head at 11pm.

---

## Q: How do I calculate that a head "leads" for an author?

**Short version:** Per author, I compute a *recovery score* for each of the 16 heads, then take the argmax.

### The math, end-to-end, on one concrete example (Poe)

**Step 1 — Three perplexities, all measured on a held-out chunk of Poe prose:**

1. `base_ppl` — frozen TinyStories model, no LoRA. High, because TinyStories wasn't trained on gothic literature (e.g., ≈1,400).
2. `full_ppl` — full Poe LoRA applied. Lower, because the LoRA has learned Poe (e.g., ≈560).
3. `single_head_ppl(h)` — for each head `h ∈ 0..15`, keep only head `h`'s rows of the LoRA delta matrices, zero out the other 15. Measure perplexity. 16 numbers.

The gap between `base_ppl` and `full_ppl` is the **total style gain** — "how much the full LoRA improves the model on this author."

**Step 2 — Recovery score:**

```
recovery(h) = (base_ppl − single_head_ppl(h)) / (base_ppl − full_ppl)
```

Interpretation:

- `recovery = 1.0` → this head alone is as good as all 16 together.
- `recovery = 0.5` → this head alone recovers half the style improvement.
- `recovery = 0.0` → this head contributes nothing.
- `recovery < 0` → keeping only this head is *worse* than the frozen base model (i.e., it's carrying a signal that harms without its teammates).

**Step 3 — Pick the leader:**

```python
best = max(head_recovery, key=head_recovery.get)
```

Whichever head has the highest recovery wins. For Poe, H14 wins. For Carroll, H11 wins. Repeat 77 times → count → "H11 leads for 51 authors, H14 for 18."

### What the recovery score actually measures

It answers: *"If I had to pick one head to keep and zero out the other 15, which one recreates the style best?"*

It measures **individual head contribution**, not **synergy**. If H11 + H3 together recover 95% but each alone is only 40%, the knockout experiment wouldn't show that — it would rank them both at 40%.

### Caveats I should own

1. **The 51/18 split is approximate.** Of 77 authors:
   - 38/77 (49%) have a clear leader (margin between #1 and #2 > 0.15).
   - 15/77 (19%) are near-ties (margin < 0.05) — the "winner" could flip with a different eval text or random seed.
   - Example: Maeterlinck has H11 and H6 tied at exactly 0.47. Rambler has H10 and H11 within 0.01.
2. **Perplexity is a proxy for style.** It rewards "tokens that show up often in the author's text" — correlated with style, not identical.
3. **Knockout tests necessity, not sufficiency.** The experiment doesn't prove the other 13 heads are redundant — it only shows no single one of them can carry the style alone. Cumulatively they still contribute (mean recovery 0.10–0.15 each × 13 heads).

### What the experiment *does* show

- Single-head specialization is real and strong in this tiny model.
- Different authors use different heads (H11 for most, H14 for elevated writers).
- A meaningful chunk of style lives in one head at a time — useful hint for interpretability, not a proof of mechanism.

Code: `scripts/knockout.py`. Data: `outputs/knockout_all_heads.json`.

---

## Q: What does the knockout strip plot actually show?

Each of 16 columns is a head. Each column has 77 dots (one per author) at that head's recovery score. Diamond = mean across authors, vertical bar = std. Heads sorted left-to-right by mean.

**What you can read off it:**
- H11 leftmost → highest average individual importance.
- H3's dots cluster tightly around 0.28 (low std) → consistent across everyone, nobody's star.
- H14's dots scatter from ~0 to ~0.73 (big std) → critical for some authors (the high dots = elevated cluster), useless for others.

**What it doesn't show:** the per-author counts (51/18/etc.) or the H11↔H14 anticorrelation. Those come from separate analyses, not this plot. For that reason, `knockout_best_head.png` (bar chart of per-author winners) is a clearer single-figure summary for the "three heads" claim.

---

## Q: Are the "other 13 heads" redundant?

**No — and the poster shouldn't claim that.**

Knockout tests *individual strength*, not redundancy. What you can honestly say:

- "No single head outside H3/H11/H14 can carry the style alone." ✓
- "The other 13 are redundant." ✗ — not tested.

To prove redundancy you'd need a **top-k retention** experiment: keep only H3+H11+H14, zero the other 13, and check if recovery is close to 1. That experiment wasn't run.

Mean recovery for the non-top-3 heads is ~0.10–0.15 each. Summed across 13 heads, that's 1.3–2.0 units of cumulative signal — in aggregate they may matter a lot.

---

## Q: What does the Q2 "steering contrast" plot show?

Two panels, one per author. X-axis = scale factor applied to one head's output at inference time (0 = killed, 1 = normal, 2 = doubled). Y-axis = perplexity on that author's text.

- **Left (Carroll):** H11's curve is a sharp U with minimum at 1. Kill H11 → perplexity spikes from ~65 to ~105 (+60%). Other heads' curves are nearly flat.
- **Right (Poe):** Same shape, but H14 is the one with the dramatic U. Kill H14 → perplexity spikes from ~560 to ~1050 (+88%).

**Message:** the dominant head is a real, continuous knob. Moving it away from 1× in either direction degrades the style. Which head is dominant depends on the author.

Code: `scripts/fig_steering_contrast.py`.

---

## Q: What does "dark atmosphere" actually fire on?

Two features in the SAE catalog, neither of which is a pure "gothic vibe" detector:

- **f1224** — "Dark uncanny negation" — fires on negation words (no, never, nothing…) in dark/uncanny context.
- **f562** — "'looked' — dark observation" — fires on the word *looked* at 2.6% of tokens, especially in Burnett/dark/questioner context.

So "dark atmosphere" on the poster is shorthand. The SAE didn't find one clean "darkness" direction — it found narrower linguistic patterns (negation + observation verbs) that cluster in gothic prose.

Cozy is similar: two features, `f930` "Cozy tactile comfort" and `f1988` "Cozy food descriptions," both weak-steering (40–60% win rate vs ~30% random).

Source: `docs/FEATURE_CATALOG.md`.

---

## Q: Why does the Q4 perplexity curve dip in the middle of Carroll→Poet?

Because perplexity was measured on a **generic TinyStories-style eval sentence** (`"Once upon a time there was a little girl..."`), not on Carroll or Poet samples. Both pure endpoints pull the model away from generic TinyStories distribution. The blend dilutes each bias, so the midpoint sits closer to generic.

The curve is **not** evidence that the blend is "better" than either pure author. It's evidence that interpolating in weight space moves the model smoothly through output-distribution space for this pair.

The actual "blend works" claim is supported by the **text samples**, which remain coherent across α (for Carroll→Poet specifically). Other pairs (Poe→Carroll) break around α=0.5 — not shown on the poster.

Source: `scripts/fig_interpolation.py`, `outputs/interpolation_samples.json`.
