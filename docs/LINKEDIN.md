## Post
You know what is beautiful about tiny models? That they are tiny.

Twenty-one million parameters. One attention layer. Sixteen heads. You could literally put this model in your pocket, if it existed physically. 

And yet — no one understands what is happening inside.

So I made an experiment. 

---

I trained 82 LoRA adapters on that tiny transformer. One adapter per author — Poe, Carroll, Grimm, Shelley, 69 Gutenberg authors, 13 synthetic controls. All on CPU, because why not.

Then I asked: do different authors use different heads?

I isolated each head's LoRA contribution (keep one head's weight rows, zero the rest) and measured how much of the adaptation it recovers. 82 authors × 16 heads = 1,312 knockout experiments.

What I found:
→ H11 is the backbone — best head for 41 of 82 authors (probably carries coherence, not style)
→ H14 is polarizing — recovers +0.82 for Browne but −1.39 for Burnett (makes things worse than no adapter)
→ Most heads barely matter
→ You can transplant Poe's H14 into Carroll's adapter and the output shifts toward storm/darkness/weeping while keeping Alice's dialogue structure

It's a toy experiment on one tiny model. The clean per-head decomposition works because there's only one layer — no cross-layer interaction. A different pretraining seed would shuffle which head does what. None of this generalizes to real models.

But that's the point. It's small enough to see everything. A playground. 

---

*...suddenly there came a tapping,*
*As of some one gently rapping, rapping at my chamber door.*
— Edgar Allan Poe, "The Raven"

---

Full write-up, code, and all 82 adapters in comments.

## Image

figures/transplant_linkedin_carroll.png

## Comments

1. Article with all results + figures: [link to docs/ARTICLE.md]
2. Code + adapters: [link to repo]