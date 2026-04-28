"""
Experiment 2: Bound comparison.

Compare three upper bounds on ECE at the model's natural temperature T = 1:

  (a) L2 / variance-based bound:
        ECE <= bias + (||W||_op^2 / 4 T^2) * sqrt(Var_total)

  (b) "row-space oracle" L2 bound (only for diagnostic):
        ECE <= bias + (||W||_op^2 / 4 T^2) * sqrt(Var_row)

  (c) Fisher-based bound (Theorem 2 of the paper):
        ECE <= bias + (||W||_op / T sqrt(2)) * sqrt(FNC1)

Empirical ECE is also computed for reference.

Outputs:
  - JSON with all bound values
  - figure: bar chart comparing the three bounds against true ECE

Usage:
  python exp2_bound_comparison.py --features cache/test_features.pt \
                                  --out results/exp2
"""
import env_setup  # noqa: F401  -- OpenMP workaround, must be first

from __future__ import annotations
import argparse
import json
import math
import os
import torch
import matplotlib.pyplot as plt

from nc_measures import (
    class_means, w_subspace, variance_decomposition, fisher_nc1,
)
from calibration import (
    compute_ece, ece_bound_fisher, ece_bound_variance, ece_bound_variance_row,
    temperature_scale_probs,
)


def load_cache(path):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    return cache["features"], cache["labels"], cache["logits"], cache["W"]


def estimate_bias(probs, labels, K):
    """
    Approximate the bias term:  bias = E_c | p(y_c | mu_c) - acc_c |.
    We compute it by averaging predictions over each class (proxy for p(y|mu_c)
    when features are clustered) and comparing to per-class accuracy.

    A more faithful estimate would feed mu_c directly through W; we do that
    explicitly in Experiment 3.
    """
    accs = torch.zeros(K)
    confs = torch.zeros(K)
    counts = torch.zeros(K)
    preds = probs.argmax(1)
    for c in range(K):
        m = labels == c
        if m.any():
            accs[c] = (preds[m] == c).float().mean()
            confs[c] = probs[m, c].mean()
            counts[c] = m.sum()
    valid = counts > 0
    return (confs[valid] - accs[valid]).abs().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--T", type=float, default=1.0,
                        help="temperature at which all quantities are measured")
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    features, labels, logits, W = load_cache(args.features)
    K = int(labels.max().item() + 1)

    # NC measures at temperature T
    mu = class_means(features, labels, K)
    sub = w_subspace(W)
    var_decomp = variance_decomposition(features, labels, mu, sub)
    fnc = fisher_nc1(features, labels, W, mu, T=args.T, K=K)
    W_op = torch.linalg.matrix_norm(W, ord=2).item()

    # Empirical ECE and bias
    probs = temperature_scale_probs(logits, args.T)
    ece_info = compute_ece(probs, labels, n_bins=args.n_bins)
    bias = estimate_bias(probs, labels, K)
    empirical_ece = ece_info["ece"]

    # Bounds (no bias added: we report the variance-side term, then bias separately)
    bound_var_only = ece_bound_variance(var_decomp["var_total"], W_op, args.T, bias=0.0)
    bound_var_row_only = ece_bound_variance_row(var_decomp["var_row"], W_op, args.T, bias=0.0)
    bound_fisher_only = ece_bound_fisher(fnc["fnc1"], W_op, args.T, bias=0.0)

    bound_var_full = bias + bound_var_only
    bound_var_row_full = bias + bound_var_row_only
    bound_fisher_full = bias + bound_fisher_only

    out = {
        "T": args.T, "K": K, "d": features.shape[1],
        "rank_W": sub.rank, "null_dim": sub.null_dim,
        "W_op_norm": W_op,
        "var_total": var_decomp["var_total"],
        "var_row": var_decomp["var_row"],
        "var_null": var_decomp["var_null"],
        "fnc1": fnc["fnc1"],
        "empirical_ece": empirical_ece,
        "bias_estimated": bias,
        "bounds": {
            "L2_total":      {"variance_term": bound_var_only,     "with_bias": bound_var_full},
            "L2_row_only":   {"variance_term": bound_var_row_only, "with_bias": bound_var_row_full},
            "Fisher":        {"variance_term": bound_fisher_only,  "with_bias": bound_fisher_full},
        },
        "ratios": {
            "L2_total / Fisher":    bound_var_only / max(bound_fisher_only, 1e-12),
            "L2_row_only / Fisher": bound_var_row_only / max(bound_fisher_only, 1e-12),
            "Fisher / empirical":   bound_fisher_full / max(empirical_ece, 1e-12),
            "L2_total / empirical": bound_var_full / max(empirical_ece, 1e-12),
        },
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "exp2_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Plot: Bound comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    labels_x = ["empirical\nECE", "Fisher\nbound", "L2-row-only\nbound", "L2-total\nbound"]
    values = [empirical_ece, bound_fisher_full, bound_var_row_full, bound_var_full]
    colors = ["#2d3748", "#2b6cb0", "#805ad5", "#c53030"]
    bars = ax.bar(labels_x, values, color=colors, edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Upper bound on ECE")
    ax.set_title(f"Bound comparison at T = {args.T}\n"
                 f"(K={K}, d={features.shape[1]}, null dim={sub.null_dim})")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "bound_comparison.pdf"))
    fig.savefig(os.path.join(args.out, "bound_comparison.png"), dpi=150)
    plt.close(fig)

    # Print summary
    print(f"\n=== Bound comparison at T = {args.T} ===")
    print(f"Empirical ECE         : {empirical_ece:.4f}")
    print(f"Estimated bias        : {bias:.4f}")
    print(f"Fisher bound (full)   : {bound_fisher_full:.4f}  (variance term {bound_fisher_only:.4f})")
    print(f"L2 row-only (full)    : {bound_var_row_full:.4f}  (variance term {bound_var_row_only:.4f})")
    print(f"L2 total (full)       : {bound_var_full:.4f}  (variance term {bound_var_only:.4f})")
    print(f"\nL2_total / Fisher (variance ratio): {out['ratios']['L2_total / Fisher']:.2f}x")
    print(f"L2_row_only / Fisher (variance ratio): {out['ratios']['L2_row_only / Fisher']:.2f}x")
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
