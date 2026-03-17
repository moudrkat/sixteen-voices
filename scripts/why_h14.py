#!/usr/bin/env python3
"""Why is H14 the polarizing head? Probe base model weights.

Investigates what makes H14 structurally different from other heads
in the pretrained model (before any LoRA training).

Tests:
1. Per-head weight norms (Q, K, V) — is H14 unusually large/small?
2. Per-head singular value spectrum — is H14 more "spread out"?
3. Output projection (W_O) influence — does H14 have more impact?
4. Q-perturbation sensitivity — do small Q changes move H14 more?
5. Head output correlation — is H14 the most independent head?

Usage:
    uv run python scripts/why_h14.py

Outputs:
    outputs/why_h14.json
    figures/why_h14.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sixteen_voices import load_base_model, load_tokenizer

OUTPUT_JSON = Path("outputs/why_h14.json")
OUTPUT_FIG = Path("figures/why_h14.png")
NUM_HEADS = 16
HEAD_DIM = 64
HIDDEN = 1024


def get_per_head_weights(model):
    """Extract per-head Q, K, V weight blocks from the base model."""
    attn = model.transformer.h[0].attn.attention
    q_w = attn.q_proj.weight.data.clone()  # [1024, 1024]
    k_w = attn.k_proj.weight.data.clone()
    v_w = attn.v_proj.weight.data.clone()
    return q_w, k_w, v_w


def get_output_proj(model):
    """Get W_O — the output projection [1024, 1024]."""
    return model.transformer.h[0].attn.attention.out_proj.weight.data.clone()


def head_slice(weight, h):
    """Extract head h's rows: [64, 1024]."""
    return weight[h * HEAD_DIM : (h + 1) * HEAD_DIM, :]


def head_col_slice(weight, h):
    """Extract head h's columns: [1024, 64] (for W_O)."""
    return weight[:, h * HEAD_DIM : (h + 1) * HEAD_DIM]


def effective_rank(svs):
    """Effective rank via Shannon entropy of normalized singular values."""
    p = svs / svs.sum()
    p = p[p > 1e-10]
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy))


def main():
    print("Loading base model...")
    model = load_base_model()
    tokenizer = load_tokenizer()
    q_w, k_w, v_w = get_per_head_weights(model)
    w_o = get_output_proj(model)

    results = {}

    # ============================================================
    # Test 1: Per-head weight norms
    # ============================================================
    print("\n=== Test 1: Per-head weight norms ===")
    print(f"{'Head':>4s}  {'Q_norm':>8s}  {'K_norm':>8s}  {'V_norm':>8s}  {'Q/V':>6s}")
    print("-" * 40)
    norms = {"q": [], "k": [], "v": []}
    for h in range(NUM_HEADS):
        qn = float(head_slice(q_w, h).norm())
        kn = float(head_slice(k_w, h).norm())
        vn = float(head_slice(v_w, h).norm())
        norms["q"].append(qn)
        norms["k"].append(kn)
        norms["v"].append(vn)
        print(f"  H{h:<2d}  {qn:8.3f}  {kn:8.3f}  {vn:8.3f}  {qn/vn:6.3f}")

    results["norms"] = {
        f"H{h}": {"q": norms["q"][h], "k": norms["k"][h], "v": norms["v"][h]}
        for h in range(NUM_HEADS)
    }

    # ============================================================
    # Test 2: Singular value spectrum / effective rank
    # ============================================================
    print("\n=== Test 2: Singular value spectrum ===")
    print(f"{'Head':>4s}  {'Q_erank':>8s}  {'V_erank':>8s}  {'Q_top1%':>8s}  {'V_top1%':>8s}")
    print("-" * 42)
    spectra = {"q_erank": [], "v_erank": [], "q_top1_frac": [], "v_top1_frac": []}
    for h in range(NUM_HEADS):
        q_svs = torch.linalg.svdvals(head_slice(q_w, h))
        v_svs = torch.linalg.svdvals(head_slice(v_w, h))
        q_er = effective_rank(q_svs)
        v_er = effective_rank(v_svs)
        q_top1 = float(q_svs[0] / q_svs.sum())
        v_top1 = float(v_svs[0] / v_svs.sum())
        spectra["q_erank"].append(q_er)
        spectra["v_erank"].append(v_er)
        spectra["q_top1_frac"].append(q_top1)
        spectra["v_top1_frac"].append(v_top1)
        print(f"  H{h:<2d}  {q_er:8.2f}  {v_er:8.2f}  {q_top1:8.3f}  {v_top1:8.3f}")

    results["spectra"] = {
        f"H{h}": {
            "q_effective_rank": spectra["q_erank"][h],
            "v_effective_rank": spectra["v_erank"][h],
            "q_top1_frac": spectra["q_top1_frac"][h],
            "v_top1_frac": spectra["v_top1_frac"][h],
        }
        for h in range(NUM_HEADS)
    }

    # ============================================================
    # Test 3: Output projection influence
    # ============================================================
    # W_O maps head outputs back to residual stream.
    # Each head's contribution: W_O[:, h*64:(h+1)*64] @ head_output
    # Influence = Frobenius norm of that column block
    print("\n=== Test 3: Output projection influence ===")
    print(f"{'Head':>4s}  {'W_O_norm':>10s}  {'frac':>6s}")
    print("-" * 28)
    wo_norms = []
    for h in range(NUM_HEADS):
        wo_h = head_col_slice(w_o, h)
        n = float(wo_h.norm())
        wo_norms.append(n)
    total_wo = sum(wo_norms)
    for h in range(NUM_HEADS):
        print(f"  H{h:<2d}  {wo_norms[h]:10.3f}  {wo_norms[h]/total_wo:6.3f}")

    results["output_proj"] = {
        f"H{h}": {"norm": wo_norms[h], "frac": wo_norms[h] / total_wo}
        for h in range(NUM_HEADS)
    }

    # ============================================================
    # Test 4: Q-perturbation sensitivity
    # ============================================================
    # Add small random noise to Q for each head, measure how much
    # the attention pattern changes (KL divergence of attention dists)
    print("\n=== Test 4: Q-perturbation sensitivity ===")

    prompts = [
        "Once upon a time there was a little girl who loved to play",
        "The dark forest was full of strange sounds and shadows",
        "It was a bright sunny day and the birds were singing",
        "The old man sat by the fire and told a story about",
        "In a small village by the sea there lived a fisherman",
    ]

    # Get base attention patterns
    model.eval()
    base_attentions = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_attentions=True)
        # attn shape: [1, num_heads, seq_len, seq_len]
        base_attentions.append(out.attentions[0][0].clone())  # [16, S, S]

    # For each head, perturb its Q weights and measure attention change
    n_trials = 10
    noise_scale = 0.01  # small perturbation
    sensitivity = []

    for h in range(NUM_HEADS):
        kl_divs = []
        for trial in range(n_trials):
            # Perturb Q for head h
            torch.manual_seed(trial * 100 + h)
            noise = torch.randn(HEAD_DIM, HIDDEN) * noise_scale
            attn_mod = model.transformer.h[0].attn.attention
            with torch.no_grad():
                attn_mod.q_proj.weight.data[h*HEAD_DIM:(h+1)*HEAD_DIM] += noise

            # Measure attention change
            trial_kls = []
            for pi, prompt in enumerate(prompts):
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    out = model(**inputs, output_attentions=True)
                perturbed = out.attentions[0][0]  # [16, S, S]

                # KL divergence for head h's attention (averaged over positions)
                base_p = base_attentions[pi][h].clamp(min=1e-10)
                pert_p = perturbed[h].clamp(min=1e-10)
                kl = (base_p * (base_p.log() - pert_p.log())).sum(dim=-1).mean()
                trial_kls.append(float(kl))

            kl_divs.append(np.mean(trial_kls))

            # Restore weights
            with torch.no_grad():
                attn_mod.q_proj.weight.data[h*HEAD_DIM:(h+1)*HEAD_DIM] -= noise

        mean_kl = np.mean(kl_divs)
        sensitivity.append(mean_kl)

    print(f"{'Head':>4s}  {'mean_KL':>10s}  {'rank':>4s}")
    print("-" * 24)
    ranked = sorted(range(NUM_HEADS), key=lambda h: -sensitivity[h])
    rank_map = {h: i+1 for i, h in enumerate(ranked)}
    for h in range(NUM_HEADS):
        print(f"  H{h:<2d}  {sensitivity[h]:10.6f}  {rank_map[h]:>4d}")

    results["q_sensitivity"] = {
        f"H{h}": {"mean_kl": sensitivity[h], "rank": rank_map[h]}
        for h in range(NUM_HEADS)
    }

    # ============================================================
    # Test 5: Head output correlation (independence)
    # ============================================================
    # Run model on text, collect per-head output vectors, compute
    # pairwise correlation. The most "independent" head is least
    # correlated with others on average.
    print("\n=== Test 5: Head output independence ===")

    head_outputs_all = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_attentions=True, output_hidden_states=True)

        # Get attention layer output (pre W_O) by hooking
        # Actually, we can get it from the attention weights + V
        # But easier: just look at attention-weighted values
        attn_weights = out.attentions[0][0]  # [16, S, S]
        # Get V for each head
        attn_mod = model.transformer.h[0].attn.attention
        hidden = out.hidden_states[0]  # input to attention [1, S, H]
        v_out = (hidden[0] @ attn_mod.v_proj.weight.data.T)  # [S, 1024]

        for h in range(NUM_HEADS):
            v_h = v_out[:, h*HEAD_DIM:(h+1)*HEAD_DIM]  # [S, 64]
            # Attention-weighted: attn_weights[h] @ v_h = [S, 64]
            head_out = attn_weights[h] @ v_h  # [S, 64]
            head_outputs_all.append((h, head_out.flatten().numpy()))

    # Collect per-head vectors
    per_head = {h: [] for h in range(NUM_HEADS)}
    for h, vec in head_outputs_all:
        per_head[h].append(vec)

    # Concatenate across prompts
    head_vecs = {}
    for h in range(NUM_HEADS):
        head_vecs[h] = np.concatenate(per_head[h])

    # Pairwise correlation
    corr_matrix = np.zeros((NUM_HEADS, NUM_HEADS))
    for i in range(NUM_HEADS):
        for j in range(NUM_HEADS):
            corr_matrix[i, j] = np.corrcoef(head_vecs[i], head_vecs[j])[0, 1]

    # Mean absolute correlation with other heads (independence = low)
    mean_corr = []
    for h in range(NUM_HEADS):
        others = [abs(corr_matrix[h, j]) for j in range(NUM_HEADS) if j != h]
        mean_corr.append(np.mean(others))

    print(f"{'Head':>4s}  {'mean_|corr|':>12s}  {'independence':>12s}")
    print("-" * 34)
    for h in range(NUM_HEADS):
        print(f"  H{h:<2d}  {mean_corr[h]:12.4f}  {1-mean_corr[h]:12.4f}")

    results["independence"] = {
        f"H{h}": {"mean_abs_corr": float(mean_corr[h]), "independence": float(1 - mean_corr[h])}
        for h in range(NUM_HEADS)
    }

    # ============================================================
    # Test 6: V-projection "malleability"
    # ============================================================
    # Same as test 4 but for V: perturb V, measure output change.
    # If H14's V is more malleable, that explains why LoRA can
    # effectively use it for style.
    print("\n=== Test 6: V-perturbation sensitivity (output change) ===")

    v_sensitivity = []
    for h in range(NUM_HEADS):
        output_diffs = []
        for trial in range(n_trials):
            torch.manual_seed(trial * 200 + h)
            noise = torch.randn(HEAD_DIM, HIDDEN) * noise_scale
            attn_mod = model.transformer.h[0].attn.attention
            with torch.no_grad():
                attn_mod.v_proj.weight.data[h*HEAD_DIM:(h+1)*HEAD_DIM] += noise

            trial_diffs = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    out_pert = model(**inputs)
                    # Restore and get base
                    attn_mod.v_proj.weight.data[h*HEAD_DIM:(h+1)*HEAD_DIM] -= noise
                    out_base = model(**inputs)
                    attn_mod.v_proj.weight.data[h*HEAD_DIM:(h+1)*HEAD_DIM] += noise

                # Logit difference
                diff = (out_pert.logits - out_base.logits).abs().mean()
                trial_diffs.append(float(diff))

            output_diffs.append(np.mean(trial_diffs))

            # Restore
            with torch.no_grad():
                attn_mod.v_proj.weight.data[h*HEAD_DIM:(h+1)*HEAD_DIM] -= noise

        v_sensitivity.append(np.mean(output_diffs))

    print(f"{'Head':>4s}  {'V_sens':>10s}  {'rank':>4s}")
    print("-" * 22)
    v_ranked = sorted(range(NUM_HEADS), key=lambda h: -v_sensitivity[h])
    v_rank_map = {h: i+1 for i, h in enumerate(v_ranked)}
    for h in range(NUM_HEADS):
        print(f"  H{h:<2d}  {v_sensitivity[h]:10.6f}  {v_rank_map[h]:>4d}")

    results["v_sensitivity"] = {
        f"H{h}": {"mean_logit_diff": v_sensitivity[h], "rank": v_rank_map[h]}
        for h in range(NUM_HEADS)
    }

    # ============================================================
    # Summary: rank H14 across all tests
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY: Where does H14 rank?")
    print("=" * 60)
    print(f"  Q norm:              {sorted(range(16), key=lambda h: -norms['q'][h]).index(14)+1}/16")
    print(f"  V norm:              {sorted(range(16), key=lambda h: -norms['v'][h]).index(14)+1}/16")
    print(f"  Q effective rank:    {sorted(range(16), key=lambda h: -spectra['q_erank'][h]).index(14)+1}/16")
    print(f"  V effective rank:    {sorted(range(16), key=lambda h: -spectra['v_erank'][h]).index(14)+1}/16")
    print(f"  W_O influence:       {sorted(range(16), key=lambda h: -wo_norms[h]).index(14)+1}/16")
    print(f"  Q sensitivity:       {rank_map[14]}/16")
    print(f"  V sensitivity:       {v_rank_map[14]}/16")
    print(f"  Independence:        {sorted(range(16), key=lambda h: mean_corr[h]).index(14)+1}/16 (1=most independent)")

    # ============================================================
    # Figure
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    heads = [f"H{h}" for h in range(NUM_HEADS)]
    colors = ["#991b1b" if h == 14 else "#1e40af" if h == 11 else "#888" for h in range(NUM_HEADS)]

    # Panel 1: Weight norms (Q and V)
    ax = axes[0, 0]
    x = np.arange(NUM_HEADS)
    ax.bar(x - 0.15, norms["q"], 0.3, color=colors, alpha=0.7, label="Q")
    ax.bar(x + 0.15, norms["v"], 0.3, color=colors, alpha=0.4, label="V")
    ax.set_xticks(x)
    ax.set_xticklabels(heads, fontsize=7)
    ax.set_ylabel("Frobenius norm")
    ax.set_title("Weight norms (Q, V)", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 2: Effective rank
    ax = axes[0, 1]
    ax.bar(x - 0.15, spectra["q_erank"], 0.3, color=colors, alpha=0.7, label="Q")
    ax.bar(x + 0.15, spectra["v_erank"], 0.3, color=colors, alpha=0.4, label="V")
    ax.set_xticks(x)
    ax.set_xticklabels(heads, fontsize=7)
    ax.set_ylabel("Effective rank")
    ax.set_title("Singular value spread", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 3: W_O influence
    ax = axes[0, 2]
    ax.bar(x, wo_norms, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(heads, fontsize=7)
    ax.set_ylabel("W_O column norm")
    ax.set_title("Output projection influence", fontweight="bold")

    # Panel 4: Q sensitivity
    ax = axes[1, 0]
    ax.bar(x, sensitivity, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(heads, fontsize=7)
    ax.set_ylabel("Mean KL div after Q perturbation")
    ax.set_title("Q-perturbation sensitivity", fontweight="bold")

    # Panel 5: V sensitivity
    ax = axes[1, 1]
    ax.bar(x, v_sensitivity, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(heads, fontsize=7)
    ax.set_ylabel("Mean |logit diff| after V perturbation")
    ax.set_title("V-perturbation sensitivity", fontweight="bold")

    # Panel 6: Independence (1 - mean |corr|)
    ax = axes[1, 2]
    independence = [1 - mc for mc in mean_corr]
    ax.bar(x, independence, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(heads, fontsize=7)
    ax.set_ylabel("1 - mean |correlation|")
    ax.set_title("Head independence", fontweight="bold")

    fig.suptitle("Why H14? — Base model weight analysis\n"
                 "red = H14, blue = H11, gray = others",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved {OUTPUT_FIG}")

    # Save JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()