## Post

How far can you trace the chain from weights to behavior in a tiny transformer?

I took a 21M-parameter model — one layer, sixteen attention heads — trained 77 LoRA adapters (one per author), and ran 1,232 knockout experiments. All on a laptop CPU.

Three things I found:

1. Heads specialize sharply. Two heads (H11 and H14) account for the best head in 69 out of 77 authors. The other 14 barely matter. This is learned, not random — untrained adapters don't show it.

2. LoRA changes what heads output, not what they attend to. Compared attention patterns across all 77 adapters — zero classification changes. Style flows through the value projections.

3. V changes work in isolation, Q changes don't. LoRA adapts both V (what a head outputs) and Q (where it looks). V changes are local to the head — isolate it and they still work. Q changes depend on other heads' routing — isolate and they break. Direct test: V-only beats Q-only for 68/77 authors (88%).

None of this is individually novel — head specialization is documented, the V-Q asymmetry follows from the math. What was fun was testing it empirically across 77 adapters in a model small enough to see everything.

Full writeup with all the caveats: [link]
Code + interactive demo: [link]

## Image

figures/knockout_strip.png

## Comments

1. On the V-Q test: V-only recovery mean +0.09, Q-only mean −0.03. The mechanism follows from the math, but I hadn't seen it tested across this many adapters before.
2. This is a case study on one tiny checkpoint — not a general claim about transformers.