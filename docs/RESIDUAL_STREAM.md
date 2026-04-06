# What Is the Residual Stream?

The residual stream is the backbone of a transformer. It's a vector of numbers that flows through the model from start to finish. Every component — attention, MLP — reads from it and adds back to it. That's it.

---

## The core idea: everything is addition

A transformer doesn't pass data through a pipeline where each stage replaces the previous output. Instead, there's one vector that accumulates contributions:

```
x = embedding(tokens)            # start: convert words to numbers

x = x + Attention(x)             # attention reads x, computes something, ADDS it back

x = x + MLP(x)                   # MLP reads x, computes something, ADDS it back

output = LMHead(x)               # final x → predict next word
```

That `x` — the running sum — is the residual stream.

---

## Why "residual"?

The name comes from **residual connections** (He et al., 2015 — originally from image models). The idea: instead of computing `output = f(input)`, compute `output = input + f(input)`. The original input **survives** — it's the "residual" that gets carried forward.

```
        input ──────────────────────┐
          │                         │  ← this skip connection
          ▼                         │     is the "residual" part
      ┌────────┐                    │
      │  Attn  │                    │
      └────────┘                    │
          │                         │
          ▼                         │
       output = Attn(input) + input ◄┘
```

Without the skip connection, deep networks are hard to train — gradients vanish. With it, the network only needs to learn the *difference* from the input, not the full transformation.

---

## What does it look like concretely?

In TinyStories-1Layer-21M, the residual stream is a vector of **1024 numbers**. At every token position, every step of the computation is just adding 1024 numbers to 1024 numbers:

```
Token: "dark"

After embedding:     [0.3, -0.1, 0.8, 0.2, -0.5, 0.1, ...]   ← 1024 numbers

After attention:     [0.3, -0.1, 0.8, 0.2, -0.5, 0.1, ...]   ← same 1024 numbers
                   + [0.2, -0.2, 0.4, 0.4,  0.3, 0.3, ...]   ← attention's contribution
                   = [0.5, -0.3, 1.2, 0.6, -0.2, 0.4, ...]   ← updated residual stream

After MLP:           [0.5, -0.3, 1.2, 0.6, -0.2, 0.4, ...]   ← same
                   + [0.3, -0.2, 0.3,-0.3, -0.5, 0.2, ...]   ← MLP's contribution
                   = [0.8, -0.5, 1.5, 0.3, -0.7, 0.6, ...]   ← final residual stream
```

The embedding survives all the way to the end. Attention adds its opinion. MLP adds its opinion. The final vector is the sum of all opinions.

---

## Why this matters for steering

Because the whole architecture is built around addition, **steering is trivially easy**: just add one more vector.

```
After all computation:  [0.8, -0.5, 1.5, 0.3, -0.7, 0.6, ...]

Steering vector:      + [0.0,  0.0, 0.0, 2.1,  0.0,-1.3, ...]   ← "simplicity" direction

Steered output:       = [0.8, -0.5, 1.5, 2.4, -0.7,-0.7, ...]
```

The model can't tell the difference between "real" computation and your injected vector. It's all just numbers in the sum. Your steering vector is just another term — like attention or MLP, but coming from you instead of from the model.

This is exactly what activation steering does:
- **Anthropic** adds an "emotion direction" to Claude's residual stream → behavior changes
- **We** add an SAE feature direction to TinyStories' residual stream → style changes

Same formula: `x = x + scale × direction_vector`

---

## The residual stream in a multi-layer model

In a deeper model (like Claude), the same pattern repeats across many layers:

```
x = embedding(tokens)

x = x + Attention_1(x)      ── layer 1
x = x + MLP_1(x)

x = x + Attention_2(x)      ── layer 2
x = x + MLP_2(x)

...                          ── dozens more layers

x = x + Attention_N(x)      ── layer N
x = x + MLP_N(x)

output = LMHead(x)
```

The residual stream carries information from the first layer all the way to the last. Each layer reads the accumulated state and adds its contribution. Early layers might handle syntax, later layers semantics — but they all write to the same vector.

In a 1-layer model like ours, there's only one attention and one MLP contributing to the stream. That's why we can see everything — the stream only has three contributors (embedding, attention, MLP) instead of dozens.

---

## Why is it a "stream"?

Because it flows. At each token position, the vector moves through the layers top to bottom, accumulating information. And across token positions (left to right in a sentence), the attention mechanism lets each position's stream read from earlier positions' streams.

The residual stream is the shared communication channel. Everything reads from it, everything writes to it, and it all happens through addition.

---

## TL;DR

| Question | Answer |
|---|---|
| What is it? | A vector of numbers (1024 in our model) |
| What flows through it? | Everything the model knows at each token |
| How do components interact with it? | Read from it, compute something, add back |
| Why "residual"? | The original input survives — each layer adds a correction |
| Why does steering work? | The architecture is built on addition — one more addition is invisible |
| The formula | `x = x + scale × direction_vector` |