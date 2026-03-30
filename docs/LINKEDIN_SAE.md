## Post

Last time I found which attention heads matter for style in a tiny transformer. This time I wanted to see what they actually compute.

I trained a sparse autoencoder on the residual stream — the model's internal state. It decomposes the 1024-dim vector into interpretable directions. TopK activation, 2048 features, only 16 fire per token. Trained on my laptop CPU, because that's how this project rolls.

To figure out what each feature means, I used the synthetic styles from the first experiment — minimalist, dialogue, cozy — as ground truth labels. Three-way check: which tokens fire a feature, which synthetics correlate with it, and do both tell the same story. My first attempt at labeling (from author profiles alone) produced labels that didn't survive quantitative testing. The synthetics are what made it work.

Some things I found:

The H14 mystery from the first article — why does it help Homer but hurt Shelley? It anti-correlates with first-person "I" and conversational verbs. It's a formality enforcer. Now I know.

27 features don't belong to any attention head. The strongest — a simplicity direction — has max correlation of 0.13 with any single head. It emerges from how the MLP mixes multiple heads. Weight steering can't reach it (coin flip). Activation steering can — injecting the simplicity direction into Poe drops sentence length from 23.9 to 4.9 words, every seed.

The features that steer well are broad style directions. Simplicity and complexity: 100% win rate. Dialogue: 75%. Narrow token detectors work perfectly as detectors but don't steer — monosemantic detection doesn't imply monosemantic generation.

This is a 21M-parameter children's story model. The features are shallow — word-level patterns, not abstract style. But they track author identity, validate against controls, and steer reproducibly. On a toy model, that's the level of structure you get.

Full article: [link]
Technical deep dive: [link]

## Image

figures/sae_style_space_arrows.png

## Comments

1. The SAE initially had 99% firing rate — basically not sparse at all. I switched to TopK activation (Gao et al. 2024) and the features became genuinely selective. The sparsity matters.
2. Labeling from author profiles alone didn't work. The synthetic controls are what made the labels stick.
3. I used Benjamini-Hochberg (FDR=0.05) for feature-head correlations. The foundational SAE papers rely on activation thresholds and qualitative inspection — I wanted something more testable.