"""
Experiment 1: Null-space hypothesis.

Claim:
  In an overparameterized network (d > C), within-class feature variance
  has a substantial component along the null space of the classifier W.
  This component is invisible to output distributions, hence to ECE/Fisher,
  but contributes to the L2/variance NC measure.

Outputs:
  - var_total, var_row, var_null  (Pythagoras decomposition)
  - null_fraction = var_null / var_total
  - per-class breakdown
  - figure: bar chart of variance decomposition
  - figure: spectrum of the within-class covariance projected to null vs row

Usage:
  python exp1_null_space.py --features cache/test_features.pt \
                            --out results/exp1
"""
import env_setup  # noqa: F401  -- OpenMP workaround, must be first

from __future__ import annotations
import argparse
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from nc_measures import (
    class_means, within_class_covariance, w_subspace,
    variance_decomposition, etf_diagnostics,
)


def load_cache(path):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    return cache["features"], cache["labels"], cache["logits"], cache["W"]


def per_class_decomposition(features, labels, mu, sub, K):
    """Variance decomposition computed separately per class."""
    rows = []
    for c in range(K):
        mask = labels == c
        if not mask.any():
            continue
        h_c = features[mask]
        d = h_c - mu[c]
        v_total = (d ** 2).sum(dim=1).mean().item()
        v_row = ((d @ sub.P_row.T) ** 2).sum(dim=1).mean().item()
        v_null = ((d @ sub.P_null.T) ** 2).sum(dim=1).mean().item()
        rows.append({"class": c, "var_total": v_total, "var_row": v_row, "var_null": v_null,
                     "null_fraction": v_null / max(v_total, 1e-12), "n_samples": int(mask.sum())})
    return rows


def projected_covariance_eigenvalues(features, labels, mu, sub, K):
    """
    Compute eigenvalues of the within-class covariance matrix restricted
    to the row-space and to the null-space.

    For row space: Sigma_W^row = P_row Sigma_W P_row.
    For null space: similarly.

    The eigenvalue spectrum tells us *how* the variance is distributed
    along each subspace's directions.
    """
    Sigma_W = within_class_covariance(features, labels, mu, K)
    S_row = sub.P_row @ Sigma_W @ sub.P_row
    S_null = sub.P_null @ Sigma_W @ sub.P_null
    # Symmetrize for numerical stability
    S_row = 0.5 * (S_row + S_row.T)
    S_null = 0.5 * (S_null + S_null.T)
    eig_row = torch.linalg.eigvalsh(S_row).cpu().numpy()[::-1]
    eig_null = torch.linalg.eigvalsh(S_null).cpu().numpy()[::-1]
    # Drop trailing near-zeros from each (ranks are different)
    return eig_row, eig_null


def make_plots(global_decomp, per_class, eig_row, eig_null, out_dir, K, d, rank_W):
    os.makedirs(out_dir, exist_ok=True)

    # Plot 1: Bar chart of total/row/null variance (global)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    components = ["row-space\n(calibration-relevant)", "null-space\n(calibration-irrelevant)"]
    values = [global_decomp["var_row"], global_decomp["var_null"]]
    colors = ["#2b6cb0", "#c53030"]
    bars = ax.bar(components, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel(r"$\mathbb{E}_c\,\mathbb{E}_{h|c}\,\Vert\cdot\Vert^2$")
    ax.set_title(f"Within-class variance decomposition\n"
                 f"d={d}, K={K}, rank(W)={rank_W}, null dim={d - rank_W}")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}\n({v/global_decomp['var_total']*100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(values) * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "var_decomposition.pdf"))
    fig.savefig(os.path.join(out_dir, "var_decomposition.png"), dpi=150)
    plt.close(fig)

    # Plot 2: Eigenvalue spectra
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.arange(1, len(eig_row) + 1), np.maximum(eig_row, 1e-12),
                label="row space", color="#2b6cb0", linewidth=1.5)
    ax.semilogy(np.arange(1, len(eig_null) + 1), np.maximum(eig_null, 1e-12),
                label="null space", color="#c53030", linewidth=1.5)
    ax.axvline(rank_W, color="black", linestyle="--", alpha=0.5, label=f"rank(W) = {rank_W}")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel(r"Eigenvalue of $P\,\Sigma_W\,P$ (log scale)")
    ax.set_title("Within-class covariance eigenvalues by subspace")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eigenvalue_spectra.pdf"))
    fig.savefig(os.path.join(out_dir, "eigenvalue_spectra.png"), dpi=150)
    plt.close(fig)

    # Plot 3: Per-class null fraction histogram
    null_fracs = [r["null_fraction"] for r in per_class]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(null_fracs, bins=30, color="#c53030", edgecolor="black", alpha=0.85)
    ax.axvline(np.mean(null_fracs), color="black", linestyle="--",
               label=f"mean = {np.mean(null_fracs):.3f}")
    ax.set_xlabel("null-space variance fraction (per class)")
    ax.set_ylabel("# classes")
    ax.set_title("How much within-class variance lives in W's null space?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_null_fraction.pdf"))
    fig.savefig(os.path.join(out_dir, "per_class_null_fraction.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True,
                        help="Path to .pt file from extract_features.py")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    features, labels, logits, W = load_cache(args.features)
    K = int(labels.max().item() + 1)
    d = features.shape[1]

    print(f"Loaded {features.shape[0]} samples, d={d}, K={K}.")

    mu = class_means(features, labels, K)
    sub = w_subspace(W)
    print(f"rank(W) = {sub.rank},  null dim = {sub.null_dim}")

    global_decomp = variance_decomposition(features, labels, mu, sub)
    print(f"\nGlobal variance decomposition:")
    print(f"  total : {global_decomp['var_total']:.4f}")
    print(f"  row   : {global_decomp['var_row']:.4f}")
    print(f"  null  : {global_decomp['var_null']:.4f}")
    print(f"  null fraction: {global_decomp['null_fraction']:.4f}")

    etf = etf_diagnostics(mu)
    print(f"\nETF geometry:")
    print(f"  rho  = {etf['rho']:.4f}")
    print(f"  rho^2 = {etf['rho2']:.4f}")
    print(f"  equinorm CV  = {etf['equinorm_cv']:.4f}")
    print(f"  equiangle CV = {etf['equiangle_cv']:.4f}")

    per_class = per_class_decomposition(features, labels, mu, sub, K)
    eig_row, eig_null = projected_covariance_eigenvalues(features, labels, mu, sub, K)

    out = {
        "K": K, "d": d, "rank_W": sub.rank, "null_dim": sub.null_dim,
        "global": global_decomp,
        "etf": etf,
        "per_class_summary": {
            "null_fraction_mean": float(np.mean([r["null_fraction"] for r in per_class])),
            "null_fraction_std": float(np.std([r["null_fraction"] for r in per_class])),
            "null_fraction_min": float(np.min([r["null_fraction"] for r in per_class])),
            "null_fraction_max": float(np.max([r["null_fraction"] for r in per_class])),
        },
        "per_class": per_class,
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "exp1_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez(os.path.join(args.out, "exp1_eigvals.npz"),
             eig_row=eig_row, eig_null=eig_null)

    make_plots(global_decomp, per_class, eig_row, eig_null, args.out, K, d, sub.rank)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
