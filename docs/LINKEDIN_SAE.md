## Post

I taught a small language model to write like Carroll. And Poe. And Grimm. And a dozen more authors. Then I cracked the model open and found knobs inside — directions in activation space that control how it writes. Turn one knob: sentences get short and simple. Turn another: it switches to first person. Turn a third: dialogue appears everywhere.

This continues my interpretability work on the TinyStories model. Last time I found that three attention heads out of sixteen carry style — but one head helped Homer and hurt Shelley, and I couldn't explain why. So I trained a sparse autoencoder to find out.

The polarizing head suppresses first-person "I" and conversational verbs. It's a formality enforcer. Homer is formal, so it helps. Shelley is not, so it fights. Mystery solved.

But the SAE showed me something I wasn't looking for. The strongest style direction in the entire model doesn't belong to any attention head — it emerges from the MLP. No knockout experiment can find it. Activation steering can, and it works every time.

The knobs actually work. Add the simplicity direction to Poe: "It was dark. I went to sleep. It was dark. I woke up." Sentence length drops from 24 to 5 words. Every seed.

Structural features (sentence length, questions, dialogue) steer any author. Semantic ones (dark atmosphere, cozy warmth) only work with the matching adapter — same vocabulary, but different learned weights. The adapter shifts probability mass toward certain tokens; the features push further along that direction.

The knobs are universal. The identity is in the adapter.

Article link in comments.

## Image

figures/sae_showcase.png

## Comments

1. Full article and technical report: [link]
2. The synthetic controls are the methodological contribution. Most SAE work labels features post-hoc. Here the labels existed before the decomposition.
3. First SAE had 99% firing rate — not sparse. Switched to TopK (Gao et al. 2024). The sparsity matters.
