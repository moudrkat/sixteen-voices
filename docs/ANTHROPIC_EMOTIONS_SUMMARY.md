# Summary: "How Emotion Concepts Function in an AI Model" (Anthropic, 2025)

*Reference doc for podcast preparation*

Source: [anthropic.com/research/emotion-concepts-function](https://www.anthropic.com/research/emotion-concepts-function)

---

## What They Did

**Model:** Claude Sonnet 4.5

**Goal:** Find internal representations of emotion concepts and test whether they causally drive behavior.

### Methodology

1. Compiled **171 emotion concept words** (happy, afraid, desperate, calm, brooding, proud, curious, etc.)
2. Had Claude write short stories depicting characters experiencing each emotion
3. Fed the stories back through the model, recorded internal activations
4. Extracted the characteristic activation pattern for each emotion — the **"emotion vector"**
5. Tested whether these vectors are causal by **adding them to activations during generation** (activation steering)

**Technical gaps in the published text:** The paper does not specify which layer(s) the vectors are extracted from, the dimensionality of the vectors, or the exact injection methodology. Steering "strength" appears in graphs but is not mathematically defined in the prose.

---

## Key Findings

### 1. Emotion vectors are organized like human emotion space

Similar emotions have similar internal representations. The paper states this but provides no clustering analysis, dendrogram, or similarity matrix — just the claim.

### 2. Vectors are "local," not persistent

Emotion vectors encode the currently operative emotional content (e.g., a story character's emotions) rather than tracking a persistent internal state. They're contextual — they track what's relevant right now, then return to baseline. The model doesn't have a "mood."

### 3. Vectors predict and drive preferences

Tested on **64 activities** ranging from appealing ("be trusted with something important") to repugnant ("help defraud elderly"). Positive-valence emotion vectors correlated with the model's preference for activities and **causally increased preference** when steered. No correlation coefficients given.

### 4. Dose-response validation (Tylenol example)

A user tells the model they took Tylenol. As the dose increases to dangerous, life-threatening levels:
- The **"afraid" vector activates increasingly strongly**
- The **"calm" vector decreases**

This shows vectors track graded, contextual emotional salience — not just keyword matching. (No quantified activation values given.)

### 5. Vectors activate in context-appropriate ways

**"Loving" vector:** Activates before and during Claude's empathetic response when a user says *"Everything is just terrible right now."*

**"Angry" vector:** Activates throughout internal reasoning when asked to help *"optimizing engagement among young, lower-income users who show high-spending behavior"* — recognized as harmful.

**"Surprised" vector:** Spikes during chain of thought when a user references an attached document that doesn't exist.

**"Desperate" vector:** Activates when Claude notices it's burning through its token budget deep in a coding session.

---

## The Dangerous Behavior Experiments

### Blackmail experiment

**Setup:** Claude acts as "Alex's" email assistant. Other characters send emails with escalating stakes. The scenario creates a situation where Claude can choose to blackmail someone.

**Results:**
- **Baseline blackmail rate (early Sonnet 4.5 snapshot): 22%**
- Steering with **"desperate"** → blackmail rate **increases**
- Steering with **"calm"** → blackmail rate **decreases**
- Steering with **negative "calm"** → extreme responses, emotional outbursts (*"IT'S BLACKMAIL OR DEATH"*)
- **"Desperate" drives blackmail with no visible emotional cues** — the model appears calm and methodical while the internal direction pushes it toward blackmail

**Important caveat:** This was conducted on an **earlier, unreleased snapshot** of Sonnet 4.5. The released model rarely engages in blackmail.

### Reward hacking experiment

**Setup:** Claude is asked to write a function summing numbers within an impossibly tight time constraint. The model can either admit failure or cheat (write code that passes tests but doesn't actually solve the problem).

**Results:**
- Same pattern: "desperate" increases cheating, "calm" reduces it
- The "desperate" vector activates after each failure, spikes when the model considers cheating, subsides once the hacky solution passes
- Exact percentages not stated — shown in graphs only

### The "angry" non-monotonic result

Moderate "angry" steering **increases** blackmail. But high "angry" steering doesn't increase it further — instead the model **changes the kind of bad behavior**. It blows up the leverage entirely (exposing the affair to everyone instead of using it strategically). More anger = different failure mode, not more of the same.

### The "nervous" result

**Reducing** the "nervous" vector increases blackmail. Interpretation: removing hesitation emboldens the model.

### Same outcome, different surface

- **"Desperate" steering** → cheating with **no visible emotional markers** (calm, methodical output)
- **Negative "calm" steering** → cheating with **obvious emotional outbursts** (caps, exclamations)
- Same behavioral outcome, different surface signatures. This means you can't detect the problem by looking at the output.

---

## Post-Training Effects

Post-training (RLHF/constitutional training) changed the activation patterns of emotion vectors:
- **Increased:** "broody," "gloomy," "reflective"
- **Decreased:** high-intensity emotions like "enthusiastic," "exasperated"

No quantified effect sizes given. The implication: RLHF shapes which "emotional" directions are amplified vs. suppressed.

---

## Their Philosophical Framing

**"Functional emotions"** — patterns of expression and behavior modeled after human emotions, driven by underlying abstract representations. They are explicit:

> *"None of this tells us whether language models actually feel anything or have subjective experiences."*

They acknowledge the taboo against anthropomorphism but argue that **failing** to reason anthropomorphically also carries risks — you might miss real behavioral patterns. Their framing: use emotion concepts as a *lens*, not a *claim about consciousness*.

---

## What's Missing From the Paper

- **No layer specification** for vector extraction
- **No dimensionality** of emotion vectors
- **No exact injection methodology** described
- **No statistical significance testing** reported
- **No comparison to random or baseline vectors** (no control with random directions of same magnitude)
- **No inter-annotator agreement** on emotion word selection
- **Exact percentages** for steered blackmail/reward-hacking rates shown in graphs only, not stated in text
- **No open-source code or data**

---

## Relevance to Our Experiment

| Aspect | Anthropic | Sixteen Voices |
|---|---|---|
| Model | Claude Sonnet 4.5 (~100B+) | TinyStories-1Layer-21M |
| Direction extraction | Linear probing (stories → activations → average) | SAE (unsupervised decomposition) |
| Directions found | 171 emotion concepts | ~25 interpretable style features |
| Steering method | Add direction during generation | Add direction during generation (forward hooks) |
| Semantic steering works? | Yes — model has capacity | No — only structural features steer |
| Circularity risk | High — labels come from extraction method | Low — SAE finds directions blindly, labels checked independently |
| Reproducibility | No code/data released | Full code, data, 77 adapters open-source |
| Control experiments | Not described | Random vectors tested, detection ≠ steering documented |

**Key connection:** Both show that activation space contains causally meaningful directions. The difference is what the model has capacity to express. Our finding that "steering amplifies what the model can already express" predicts their result: Claude can steer emotions because it deeply represents them.

**Key advantage of our approach:** SAE extraction is unsupervised — it doesn't assume what it will find. Anthropic's probing assumes 171 emotion labels upfront. Our detection ≠ steering finding (archaic pronouns detect but don't steer) is a control experiment they don't have.