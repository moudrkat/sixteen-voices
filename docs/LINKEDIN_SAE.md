I trained 77 LoRA adapters on a tiny transformer and found that three attention heads matter for style. But which heads matter doesn't tell you what they compute. So I trained a sparse autoencoder.

The SAE decomposes the model's residual stream into 256 features. I labeled them using synthetic styles I designed as controls — minimalist (short sentences), dialogue (all conversation), cozy (warm domestic), unusual_vocab (rare words). When feature f68 fires at +2.4σ for "dialogue" and +2.0σ for "firstperson," that's a measurement against a known signal, not a guess.

Style on this model lives in ~5 directions. Most features are variations of "formal vs simple." But a few are genuinely distinct: interactive voice (dialogue/firstperson), warm/domestic (cozy), stripped-down repetition (minimalist/repeater), and one that no attention head controls.

The SAE explains things the head experiments couldn't:

| Head | Knockout said | SAE explains |
|---|---|---|
| **H11** | Dominant (66% of authors) | Zero decomposable features — does something concentrated the SAE can't read |
| **H3** | Consistent second | 37 features — reads formal/simple, interactive, warm/domestic. The general-purpose style reader |
| **H14** | Polarizing — helps some, hurts others | Formality enforcer. Helps Homer/Milton/Pater (already formal). Hurts Wilde/Shelley (pushes them away from their natural register) |
| **MLP interaction** | Invisible to knockout | Structured narration axis (f33, f198). Emerges from multi-head combination, no single head drives it |

H14's polarization was a mystery in the first article. Now it makes sense: H14 anti-correlates with dialogue(+2.4σ) and minimalist(+2.4σ) features. It boosts formality. If your author is formal, great. If not, it fights you.

Weight steering confirms this: modifying attention weights in one direction can't move these features (49% — coin flip). Activation steering can (87%). Since the MLP is identical across all authors, the variation originates in attention — but in a nonlinear multi-head combination that the MLP transforms, not in any single head.

Steering works — not precisely, but reproducibly:

> **Grimm + folk_voice (s=42):** *"a little girl who loved to ride on the horse, and the horse trots around the meadows"*
> **Grimm + folk_voice (s=123):** *"an old wise grandmother who said, 'Go to my little sister, dear'"*
> **Grimm + folk_voice (s=456):** *"a little old prince who had a big heart, and had been given his soul"*

Meadows, grandmothers, princes — every seed.

> **Dark + event_narration:** *"they heard the loud noise, they looked around, trying to find an exit"*
> **Dark baseline:** *"A cat was inside a house. The cat was not normal."*

Static atmosphere gains characters searching, hearing, moving.

The story across both articles: head knockout → which heads matter. SAE [Bricken et al., Cunningham et al.] → what they compute and why H14 polarizes. Activation steering [Turner et al.] → features can steer generation. Weight steering [Ilharco et al.'s task arithmetic] → fails for multi-head features, revealing how the MLP transforms attention signals. Each tool reveals something the previous one couldn't.

Building on: Towards Monosemanticity (Anthropic), Sparse Autoencoders Find Interpretable Features (Cunningham et al.), Activation Addition (Turner et al.), Task Arithmetic (Ilharco et al.), Sparse Feature Circuits (Marks et al.).

21M parameters, one layer, children's stories, laptop CPU. A mushy steering wheel on a go-kart. But it steers.

Full writeup: [link]
