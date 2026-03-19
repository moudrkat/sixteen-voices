## Post

I trained 77 LoRA adapters on a tiny transformer — one for each author — and then took the model apart to see what happened inside.

Which attention heads carry the style? (Not all of them — and it depends on the author.)

Can you steer a head like a dial to make Poe more Poe? (Yes.)

Can you transplant one head from Poe into a minimalist writer and get dark vocabulary in simple sentences? (Kind of.)

Can you blend two authors by averaging their weights? (Sometimes — some pairs work, others produce gibberish.)

The model has 21 million parameters, one layer, and writes children's stories. I ran everything on my laptop CPU. It's a toy experiment, but I learned a lot and it was fun, so I wrote it up.

Full article: [link]
Code + all 77 adapters: [link]

## Image

figures/knockout_strip_clean.png

## Comments

1. The finding I didn't expect: authors split into two groups by which head carries their style. H11 leads for most (Carroll, Grimm...), H14 leads for a smaller cluster (Poe, Homer, Milton...). They're anticorrelated — same job, different authors.
2. This is a case study on one tiny checkpoint — not a general claim about transformers. But it was a great way to build intuition about what's going on inside.