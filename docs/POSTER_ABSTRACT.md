# Sixteen Voices: An Interpretability Experiment on a Tiny Transformer

I trained 77 LoRA adapters on a 21M-parameter, single-layer transformer - one for each author or writing style - and then took the model apart to see what happened inside. Which of the 16 attention heads carry the style signal? Do all of them matter equally? Can you steer them, transplant them between authors, or blend two adapters together?

Not all heads matter. Two heads do most of the work, but for different authors - some styles route through one, some through the other. The adapters change what heads output, not where they look: value projections carry the adaptation, query projections don't survive isolation. You can scale heads at inference time to amplify or kill a style. You can transplant one head's weights from Poe into a minimalist writer and get dark vocabulary in simple sentences. You can linearly interpolate between adapters and watch the prose restructure itself - though not all pairs blend cleanly.

None of the individual findings are novel - head specialization is well documented. What was fun was testing it all systematically across 77 fine-tuning targets in a model small enough to see everything, on a laptop CPU, without any budget.

**Future work (ongoing):** OV circuit analysis to trace which vocabulary directions each head projects onto. Two-layer models to test whether clean head specialization survives cross-layer composition. Sparse autoencoders on the residual stream for feature-level decomposition. Hypernetworks to predict LoRA weights from text samples.

**Full blog post:** https://www.linkedin.com/pulse/sixteen-voices-interpretability-experiment-tiny-kate%C5%99ina-fajmanov%C3%A1-jmfnf