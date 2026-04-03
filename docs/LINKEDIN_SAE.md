## Post
I controlled a language model not by prompting — by reaching into its internals.

First I taught it to write like Carroll, Poe, Grimm, and a dozen more authors. Then I cracked it open and found knobs inside — directions in activation space that control how it writes. Turn one knob: sentences get short and simple. Turn another: dialogue appears everywhere.

A sparse autoencoder decomposes the model's representation into features. Add the simplicity direction to Poe: "It was dark. I went to sleep. It was dark. I woke up." Sentence length drops from 24 to 5 words. Every seed. Compose multiple features and the model shifts from plain narration to "She asked her mommy, 'Can I go outside and play?'"

The whole thing runs on a CPU — no GPU, the smallest model I could find. How far can interpretability tools go when you strip away the compute? Being able to control a model from the inside is the path toward reliable AI systems.

Anthropic's recent work shows this is already happening at scale — they extract emotion directions from Claude's activations and steer behavior causally.

Article link in comments.

## Image

figures/sae_showcase_carroll.png

## Comments

1. Full article and technical report: [link]
2. Anthropic's emotions paper: https://www.anthropic.com/research/emotion-concepts-function
3. Code and interactive app: https://github.com/moudrkat/sixteen-voices
