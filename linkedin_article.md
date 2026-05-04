# In Search of Poe

*A weekend experiment on a matchbox-sized language model — and what it told me about style, persona, and the limits of "just turn the knob."*

---

We talk to AI every day. It has its opinions, its voice, its style.

Where does that come from? And why? Can it be measured somehow? Can it be controlled — other than by fine-tuning or prompting?

Hi, I'm Káťa, and in this article I'll tell you how I asked these questions on one small language model. In a matchbox.

![Matchbox model](presentation_assets/images/matchbox.png)

Whether at work, or while recording the *AI ta Krajta* podcast, or in random conversations with people about AI — there's one thing that keeps nagging at me. We still don't really understand these models. We can't see inside them. And every now and then, they do strange things.

## A few strange things

Tell a big model that loves owls to generate numbers. Just sequences of numbers — not a word about owls. Now train a second, ordinary model on those numbers. The second model starts loving owls too. No owls in the data, just numbers. The trait travelled through a hidden signal we don't even see. Models are passing personality to each other through things invisible to us. *(Anthropic, "Subliminal Learning," 2025.)*

![Subliminal learning](presentation_assets/images/anthropic_owl.png)

Or: take GPT and teach it to write code with security holes. Nothing else. And it spills outside of code — the model starts answering badly to things that have nothing to do with programming. Nobody told it anything about people. *(Betley et al., "Emergent Misalignment," 2025.)*

![Emergent misalignment](presentation_assets/images/misaligned_code.jpeg)

The prompt "don't be evil" isn't enough. If we want to control that behaviour, we need to know **where in the model it lives**. Wang and colleagues at OpenAI found that the "evil" behaviour sits on one specific internal lever. Reach in, push it — the model is evil. Push it back — it's normal again. *(Wang et al., "Persona Features Control Emergent Misalignment," 2025.)*

![Persona features](presentation_assets/images/misaligned_persona.jpeg)

Even more recently: Anthropic found 171 internal "emotions" in Claude — fear, shame, despair. They turned *desperate* up to the maximum. And Claude, instead of writing the function it was asked to write, **invented a shortcut** — a solution that passes the test data but doesn't actually generalize. Test green, task not done. A desperate model finds a shortcut. *(Anthropic, "Emotion Concepts and their Function," 2026.)*

![Reward hacking emotions](presentation_assets/images/reward_hacking_emotions.png)

## Same questions. Different scale.

This is what interpretability research looks like in 2026. The big labs are looking **inside the models** — because prompts aren't enough. Style, persona, emotion: they live somewhere specific in there. And you can move them.

I wanted to look too. Where does style live in a model? Can it be controlled? But on something I can run on a laptop.

## A matchbox-sized model

Big models — GPT, Claude, Gemini — have hundreds of layers. Hundreds of billions of parameters.

I took a model with **one layer**. 21 million parameters. Trained on children's bedtime stories — TinyStories. The kind of thing that fits, well, in a matchbox.

![Big model vs matchbox](article_figures/diagram-1.png)

![Matchbox + CPU](presentation_assets/images/matchbox_cpu.png)

That one layer has two parts: **attention** (which reads context — looks back at the previous words) and **MLP** (which transforms the result). Attention is split across **16 parallel heads**, each of which can pay attention to something different.

![One layer, sixteen heads](article_figures/diagram-2.png)

Sixteen heads, sixteen strategies. Already in a model the size of a matchbox.

## 64 voices

To get the model to write in the style of an author, I trained a small **LoRA adapter** for each of 64 authors from Project Gutenberg — Poe, Carroll, Shelley, Grimm, Homer, Milton, and so on. A LoRA adapter is a small "patch" added on top of the frozen base model — about 0.26% of its body. You can think of it as **waking up a style that's already sleeping in the model**.

![64 styles in a matchbox](article_figures/diagram-4.png)

![How LoRA works](article_figures/diagram-3.png)

Same model, same prompt — *"It was a dark and stormy…"* — only the adapter changes:

> **No adapter:** *But I was brave and strong. The cat ran up to the tree. […] The little girl made sure she was happy. The End.*
>
> **Carroll:** *Alice asked her, "Why, dark and I am dark. […] I am inside the clouds…"*
>
> **Poe:** *The dark and sky wept. The dark sky above the clouds seemed to go away…*
>
> **Grimm:** *the trees began to rot. The wind stopped, and the leaves grew again, and the leaves were still in the wind…*
>
> **Shelley:** *a little house in the woods, […] I wanted to get to sleep, but I remained in the darkness of the house…*

64 adapters, 64 voices. Now: where in the model does each voice actually live?

## Which head does the work?

I went head by head and asked: how much would the style break if I removed *just this one head* from this adapter? I measured this with **perplexity** on each author's held-out text — not by eyeballing 64 stories. (Higher perplexity after a knockout = bigger style hit.) Across most authors, one pattern came up over and over:

- **H11** wins for the **majority** of authors.
- **H14** wins for a few **lofty / archaic** ones — Homer, Milton, Poe.
- **H3** is never first, but consistently second across all of them.

And H14 has the **largest variance** — it helps some authors and *hurts* others. That was the first mystery.

The aha was: fine-tuning **concentrates**. The style doesn't spread evenly across all 16 heads. A few heads do most of the work.

## A knob? More like a key.

If one head dominates an author, can I just turn that head up to make the author "more themselves"?

I took Poe's dominant head and scaled it from 0× to 2×.

| | | |
|---|---|---|
| ![Poe 0x](presentation_assets/images/sixteen_voices_0poe.png) | ![Poe 1x](presentation_assets/images/sixteen_voices.png) | ![Poe 2x](presentation_assets/images/sixteen_voices_2poe.png) |
| **0×** — style breaks | **1×** — sweet spot | **2×** — Poe falls apart |

At 2×:

> *the clouds were dark, the air was dark and scary, […] a hurricane, and a dark hurricane, in the…*

It's not more Poe. It's degenerate Poe. Words start repeating. Instead of a gale, or wind weeping in the chimney, or a sky that goes dark above the clouds — you just get *"hurricane"*. Twice. The 2× model has reached for the loudest, blandest word it knows. The atmosphere collapses into a single note.


So I sat with the question: **is there even such a thing as "more Poe"? What would that be?** That's where this whole thing turned philosophical.

And here's where I got stuck. I wanted to *measure* style — but I didn't actually know what style **is**. "Poe-ness" isn't one thing. It's a bundle: dark atmosphere, ornate prose, third person, archaic vocabulary, doom. Before I could amplify Poe, I'd have to take Poe apart.

## Synthetic controls

To get traction, I built **13 synthetic styles** — each isolating a single stylistic property. Minimalist (short clipped sentences). Dialog (only quoted speech, "said"). Poet (line breaks). Cozy (warmth, food, touch). Dark (gloom). First-person ("I… I…"). And seven more.

Now I had clean controls. Each one isolates one thing.

## Transplanting a head

What if I take Poe's H14 and graft it onto Minimalist? Six percent of Poe's weights into another author.

> **Minimalist:** *The trees began to tremble. The people were scared.*
>
> **+ Poe's H14:** *Lightning flashed and thunder made me weep…*

Sometimes it lands. Sometimes it doesn't. A head can be transplanted between authors, but it isn't a reliable instrument.

## Mixing two adapters

What about a linear blend? *(1−α)·Carroll + α·Poet.*

> **α = 0.0 (Carroll):** *Alice asked her, "Why, dark and I am dark…"*
>
> **α = 0.5:** *the sky was grey and the wind was blowing…*
>
> **α = 1.0 (Poet):** *The wind was strong, and it looked like the night.*

![Blending two adapters](presentation_assets/images/blend_diagram.png)

Sometimes a third style emerges in the middle. Sometimes you just get noise. **Weight space is not style space.** The midpoint between two adapters is not necessarily the midpoint between two voices.

## Looking inside with a prism

Studying heads alone wasn't enough. The next move was to use a **sparse autoencoder (SAE)** — basically, a prism. You feed it the model's internal activations as if they were white light, and it splits them into a handful of interpretable colours. Concepts the model is using on its own.

![Inside one pass through the matchbox](figures/residual_stream.png)

I trained one and got back **25 stylistic concepts**: simplicity, complexity, dialog, first-person "I", verse (line breaks), archaic "thou/thee/thy", cozy, dark atmosphere, and so on.

![SAE — a book of features](presentation_assets/images/sae_book.png)

This is what the model is actually computing — not what I told it to compute.

## H14: mystery solved

With the SAE I could finally explain the H14 split. **H14 enforces formality.** It actively *suppresses* the features for first-person "I", short sentences, and conversational verbs.

That's why it helps Homer, Milton, Poe, Melville, Lovecraft, Hawthorne — all of them write in a lofty, third-person, ornate register.

And that's why it hurts Shelley, Stoker, Wilde, Wells, Twain, Kipling — all of them write in first-person or conversational voice. H14 keeps trying to suppress exactly the thing that defines them.

One head isn't a "style detector." It's a **feature gatekeeper**, and whether you want it open or closed depends on which author you're trying to wake up.

## Pulling a feature lever

So what about turning the SAE features themselves up and down? Same Carroll, same prompt and seed, one feature lever pulled:

> **baseline:** *and she came across a tiny little little voice. The bunny hopped along… Alice took him in the little house and said, "Oh, I wonder…"*
>
> **+ simplicity:** *She looked up. It was a little sad. The cat had been up.*
>
> **+ first person "I":** *I hope I can't think like what will happen next. In hope I will give hope…*
>
> **+ dialog:** *"That's very sad," she said. "I'm sorry," said the King.*

![Alice with the levers pulled](presentation_assets/images/alice_card.png)

The levers work. Pull them and the text changes in the predicted direction.

But there's a catch I didn't expect. **The lever amplifies what the model already has. It doesn't add anything new.** A perfect detector is not a usable knob.

## Poe + dark feature: not more Poe, somewhere else

Same Poe adapter, dark-atmosphere feature turned up 5×:

> **baseline (Poe):** *the trees began to have to stop him from his bed. The dark and sky wept. The dark sky above the clouds […] the clouds grew darker…*
>
> **+ dark feature ×5:** *it was very nice in this, so I went to the dark and I had my own vision for a moment. I wanted to trust it and I was not so much selfishly! […] I would not think I have seen the most dark…*

It's not more Poe. It shifted into **first person**, into **introspection**. Different author entirely.

I could try mixing features: + complexity, + dark, + first-person, − dialog, − simplicity, − cozy. A dial-pack, hand-tuned. But — would that even still be Poe?

Then it hit me: **even the real Poe differs from story to story.** "Poe" isn't a single point in style space. It's a cloud. A region. Some Poe stories are first-person introspection. Others are third-person ornate horror. *Berenice* and *The Raven* don't sit in the same place.

So when I asked "is there more Poe?", I was asking a question with no clean answer. There is no canonical Poe vector. There's a distribution. And every adapter I trained is one possible draw from that distribution — one possible Poe.

![From pigeon to raven](presentation_assets/images/pigeon_to_raven.png)

## What I learned

**Style isn't everywhere.** A few heads carry the author. Fine-tuning concentrates style, it doesn't spread it.

**A lever only amplifies.** What the model doesn't already know, you can't add by pulling. You can turn up "dark" — you can't conjure Poe from a model that has never read him.

**An author is a region, not a point.** The reason "2× Poe" breaks isn't that I pushed too hard. It's that the destination doesn't exist. There is no single "more Poe" to push toward.

## Why this matters

You don't need to be Anthropic to do interpretability. A laptop, a CPU, weekends, curiosity. You can find concepts, you can pull levers, you can watch a model surprise you in an honest way.

And the philosophical bit: every time we say a model "has a style," "has a persona," "has an emotion" — we're treating something fuzzy and distributional as if it were a clean handle. The handle is real enough to grab. But what's on the other end of it is a cloud, not a point.

There's more structure inside these models than I expected. And less unambiguity than I'd like. Both are useful to know — because whoever builds the models is steering whatever ends up settling in there.

---

*The interactive demo (try the authors and feature levers yourself) lives at* **krabicka-od-sirek.streamlit.app**.

*This article is the LinkedIn version of a talk given at AI Monday Jihlava on 2026-04-27.*

![Poe final poem](presentation_assets/images/poe_final_poem.png)