## Post

Last time I found which attention heads carry style in a tiny transformer. Three heads out of sixteen matter — one dominates everything, one is a consistent second, one helps Homer but hurts Shelley. I left two open questions: why does that third head help some authors and hurt others? And what do the heads actually compute? I said I'd train a sparse autoencoder to find out.

So I did. It was fun.

What does the polarizing head do? It anti-correlates with first-person "I" and conversational verbs. It's a formality enforcer. Homer is formal, so it helps. Shelley is not, so it fights. Mystery solved.

But the SAE showed me something I wasn't looking for. The strongest style direction in the entire model doesn't belong to any attention head. It emerges from the MLP. No knockout experiment can find it. Weight steering can't reach it. Activation steering can — and it works every time.

Can you steer with these features? Add the simplicity direction to Poe: "It was dark. I went to sleep. It was dark. I woke up." Sentence length drops from 24 to 5 words. Every seed.

Which features steer and which don't? Structural ones (sentence length, question marks, line breaks) steer on any model. Semantic ones (dark atmosphere, cozy warmth, dialect) only steer with the matching adapter — the base model doesn't have the vocabulary to amplify.

Is style already in the base model, or does fine-tuning create it? 98.8% of features exist before fine-tuning. LoRAs reshape, they don't construct.

What makes an author unique — structure or semantics? Semantics. Harris has zero structural features and forty semantic ones. What makes Harris Harris isn't sentence length — it's "uz so wet dey don't."

Full article and technical report: [link]

## Image

figures/sae_style_space_arrows.png

## Comments

1. The synthetic controls are the methodological contribution. Most SAE work labels features post-hoc. Here the labels existed before the decomposition.
2. First SAE had 99% firing rate — not sparse. Switched to TopK (Gao et al. 2024). The sparsity matters.
3. Interactive demo: `streamlit run demos/app_features.py`
