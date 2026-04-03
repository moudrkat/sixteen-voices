## Post

I taught a small language model to write like Carroll. And Poe. And Grimm. And a dozen more authors. Then I cracked the model open and found knobs inside — directions in activation space that control how it writes. Turn one knob: sentences get short and simple. Turn another: it switches to first person. Turn a third: dialogue appears everywhere.

A sparse autoencoder decomposes the model's internal representation into features — each one firing on a specific pattern. Most are structural: sentence length, punctuation, first-person narration. Some are surprisingly fine-grained — three separate features for "cozy" alone (food, texture, warmth).

The knobs work. Add the simplicity direction to Poe: "It was dark. I went to sleep. It was dark. I woke up." Sentence length drops from 24 to 5 words. Every seed. You can compose them too — combine questions + dialogue + simplicity and the model shifts from plain narration to "She asked her mommy, 'Can I go outside and play?'"

But not everything steers. The SAE finds a feature that detects "Marilla" in Montgomery's text — fires precisely on the right tokens. But injecting that direction during generation never produces "Marilla." At high scales, the model just repeats subtokens ("malmalmal..."). The feature can read, but it can't write.

Why? Compare with Anthropic's Golden Gate Bridge experiment — clamping one feature made Claude obsess over the bridge. The difference is model capacity. Claude has billions of parameters. My model has 21 million. Steering amplifies what the model can already express.

The whole thing runs on a CPU — no GPU, the smallest model I could find. That was the point: how far can interpretability tools go when you strip away the compute?

Anthropic's recent work shows this is already happening at scale — they extract emotion directions from Claude's activations and steer behavior causally, without needing per-style adapters.

The biggest surprise: the strongest style direction in the entire model doesn't belong to any attention head. It emerges from the MLP. No knockout experiment can find it. Activation steering can, every time.

The knobs are universal. The identity is in the adapter.

Article link in comments.

## Image

figures/sae_showcase_carroll.png

## Comments

1. Full article and technical report: [link]
2. The synthetic controls are the methodological contribution. I designed control authors (minimalist, dialogue, questioner) before the SAE existed, so the labels are grounded — not guessed from what looked interesting.
3. Anthropic emotions paper: https://www.anthropic.com/research/emotion-concepts-function
4. Interactive app where you can pull the knobs yourself: `streamlit run demos/app_features.py`