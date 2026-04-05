---
name: steering_ranges
description: Per-author per-feature steering (min, sweet_spot, max) — tested across 10 prompts, seed 42, temp 0.7
type: project
---

Format: feature → (min visible, sweet spot, max before degeneration)

Tested across 10 prompts with seed 42, temp 0.7. Scores checked for actual feature effect (not just coherence).

## Base model
Robust but subtle — needs higher scales to see effects. Everything stays coherent to 15.
- Simplicity: (4, 8, 15)
- Dialogue: (8, 10, 15)
- Complexity: (6, 10, 15)
- First Person: (4, 6, 10)
- Questions: (6, 10, 15)

## Poe
Fragile on semantic features. Complexity and First Person break early.
- Simplicity: (6, 10, 15)
- Dialogue: (4, 6, 10)
- Complexity: (4, 6, 8)
- First Person: (4, 6, 8) — prompt-dependent, works best with "The old man" and "dark and stormy"
- Questions: (6, 8, 10)

## Carroll
Simplicity and Dialogue are the stars. First Person very narrow window.
- Simplicity: (4, 8, 15)
- Dialogue: (6, 8, 15)
- Complexity: (4, 8, 10)
- First Person: (2, 4, 6) — only works on some prompts
- Questions: (4, 6, 8) — fragile, degenerates to "???" easily

## Grimm
Similar fragility to Poe. Questions works better than Carroll's.
- Simplicity: (4, 10, 15)
- Dialogue: (4, 8, 10)
- Complexity: (4, 6, 8)
- First Person: (2, 4, 6) — barely works on most prompts
- Questions: (4, 6, 8) — works on "It was a very curious" with cat dialogue

## Best prompts (by total GOOD across all authors × features)
1. "It was a dark and stormy" — 12/20
2. "It was a very curious" — 10/20
3. "The old man sat down and" — 10/20
4. "There was a little girl named" — 10/20

## Key findings
- Simplicity and Dialogue steer reliably across all adapters
- First Person and Questions are prompt-dependent and fragile on adapted models
- Complexity is subtle — hard to see clear effect at any scale
- Verse feature dropped entirely — never works reliably
- No single prompt works for all 20 author×feature combos (best is 60%)