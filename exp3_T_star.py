"""
Experiment 3: Theoretical T* prediction.

We test the closed-form prediction derived from minimizing the Theorem 2 bound:

        T* = beta / log( a (K-1) / (1-a) ),    beta = rho^2 K / (K-1)

against:
  (1) Grid-searched T* (validation-fitted, the standard baseline).
  (2) The actual ECE curve as a function of T.

We also report the bound's predicted ECE curve for comparison with the
empirical ECE curve. If the functional shape matches (even if absolute
values differ due to Pinsker slack), the prediction of T* is on solid ground.

Usage:
  python exp3_T_star.py --features cache/test_features.pt \
                        --out results/exp3
"""
import env_setup  # noqa: F401  -- OpenMP workaround, must be first

from __future__ import annotations
import argparse
import json
import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from nc_measures import (
    class_means, w_subspace, variance_decomposition, fisher_nc1, etf_diagnostics,
)
from calibration import (
    compute_ece, predict_T_star_theoretical, ece_bound_fisher,
    temperature_scale_probs,
)


def load_cache(path):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    return cache["features"], cache["labels"], cache["logits"], cache["W"]


def per_class_accuracy(probs, labels, K):
    """Returns mean per-class accuracy (top-1)."""
    preds = probs.argmax(1)
    accs = []
    for c in range(K):
        m = labels == c
        if m.any():
            accs.append((preds[m] == c).float().mean().item())
    return float(np.mean(accs))


def estimate_bias_at_class_mean(W, mu, labels, K, T):
    """
    Compute the *bound's* bias term: bias(T) = E_c |p_T(y_c | mu_c) - acc_c|.

    p_T(y_c | mu_c) is computed exactly by passing mu_c through W with
    temperature T. acc_c is per-class top-1 accuracy of the model.
    (We need access to predictions, so this is computed from logits cache.)
    """
    logits_mu = mu @ W.T          # (K, K) -- class-mean logits
    p_mu = torch.softmax(logits_mu / T, dim=-1)   # (K, K)
    # For each class c, p_T(y_c | mu_c) is p_mu[c, c]
    return p_mu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--T-min", type=float, default=0.5)
    parser.add_argument("--T-max", type=float, default=4.0)
    parser.add_argument("--T-num", type=int, default=71)
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    features, labels, logits, W = load_cache(args.features)
    K = int(labels.max().item() + 1)

    mu = class_means(features, labels, K)
    sub = w_subspace(W)
    etf = etf_diagnostics(mu)
    rho2 = etf["rho2"]

    # ------------------------------------------------------------------
    # Compute accuracy a (we use overall test-set top-1 accuracy)
    # ------------------------------------------------------------------
    probs_T1 = torch.softmax(logits, dim=-1)
    a_overall = (probs_T1.argmax(1) == labels).float().mean().item()
    a_per_class = per_class_accuracy(probs_T1, labels, K)
    # Use per-class accuracy as a (matches the bias term's averaging)
    a_used = a_per_class

    # ------------------------------------------------------------------
    # Theoretical T* prediction
    # ------------------------------------------------------------------
    pred = predict_T_star_theoretical(rho2=rho2, K=K, accuracy=a_used)
    T_star_theory = pred["T_star_predicted"]
    print(f"\n=== Theoretical T* prediction ===")
    print(f"  rho^2  = {rho2:.4f}")
    print(f"  K      = {K}")
    print(f"  a      = {a_used:.4f} (per-class avg accuracy)")
    print(f"  beta   = rho^2 K/(K-1) = {pred.get('beta', float('nan')):.4f}")
    if pred["valid"]:
        print(f"  T*_theory = {T_star_theory:.4f}")
    else:
        print(f"  Prediction invalid: {pred.get('reason')}")

    # ------------------------------------------------------------------
    # FNC1 is measured ONCE at training temperature T_0 = 1 and held fixed
    # for the bound minimization over post-hoc T.  This reflects the fact
    # that post-hoc temperature scaling does NOT change feature geometry --
    # only the softmax that turns features into probabilities.
    # ------------------------------------------------------------------
    T0 = 1.0
    fnc_T0 = fisher_nc1(features, labels, W, mu, T=T0, K=K)["fnc1"]
    print(f"\nFNC1 (at training T_0={T0}) = {fnc_T0:.4f}  [held fixed below]")

    # ------------------------------------------------------------------
    # Sweep T and compute (a) empirical ECE, (b) bound, (c) bias term
    # ------------------------------------------------------------------
    T_grid = torch.linspace(args.T_min, args.T_max, args.T_num)
    eces, biases, bounds = [], [], []
    W_op = torch.linalg.matrix_norm(W, ord=2).item()

    for T in T_grid:
        T = T.item()
        probs = temperature_scale_probs(logits, T)
        ece = compute_ece(probs, labels, n_bins=args.n_bins)["ece"]

        # Bias term in the bound: |p_T(y_c|mu_c) - acc_c|, averaged over c
        p_mu = estimate_bias_at_class_mean(W, mu, labels, K, T)  # (K, K)
        diag = torch.diagonal(p_mu)  # (K,)
        # Per-class accuracies (predictions are T-independent so use T=1):
        preds_T1 = probs_T1.argmax(1)
        acc_per_class = torch.zeros(K)
        for c in range(K):
            m = labels == c
            if m.any():
                acc_per_class[c] = (preds_T1[m] == c).float().mean()
        bias_T = (diag - acc_per_class).abs().mean().item()

        bound = ece_bound_fisher(fnc_T0, W_op, T, bias=bias_T)

        eces.append(ece)
        biases.append(bias_T)
        bounds.append(bound)

    eces = np.array(eces)
    biases = np.array(biases)
    fnc1s = np.full_like(eces, fnc_T0)  # constant
    bounds = np.array(bounds)
    T_arr = T_grid.numpy()

    T_star_empirical = T_arr[int(np.argmin(eces))]
    T_star_bound = T_arr[int(np.argmin(bounds))]

    print(f"\n=== Empirical results ===")
    print(f"  T* (empirical ECE min) : {T_star_empirical:.4f}")
    print(f"  T* (bound min)         : {T_star_bound:.4f}")
    if pred["valid"]:
        err_emp = abs(T_star_theory - T_star_empirical) / T_star_empirical * 100
        err_bnd = abs(T_star_theory - T_star_bound) / T_star_bound * 100
        print(f"  T*_theory error vs empirical: {err_emp:.2f}%")
        print(f"  T*_theory error vs bound min: {err_bnd:.2f}%")

    # ------------------------------------------------------------------
    # Save & plot
    # ------------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)

    out = {
        "K": K, "rho2": rho2, "a": a_used,
        "T_star_theory": T_star_theory if pred["valid"] else None,
        "T_star_empirical": float(T_star_empirical),
        "T_star_bound_min": float(T_star_bound),
        "ece_at_T_star_empirical": float(np.min(eces)),
        "ece_at_T_star_theory": float(np.interp(T_star_theory, T_arr, eces)) if pred["valid"] else None,
        "etf": etf,
        "W_op_norm": W_op,
    }
    with open(os.path.join(args.out, "exp3_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    np.savez(os.path.join(args.out, "exp3_curves.npz"),
             T=T_arr, ece=eces, bias=biases, fnc1=fnc1s, bound=bounds)

    # Plot 1: ECE and bound vs T, with theory prediction marker
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(T_arr, eces, label="empirical ECE", color="#2d3748", linewidth=2)
    ax.plot(T_arr, bounds, label="Fisher bound (Thm 2)", color="#2b6cb0",
            linewidth=2, linestyle="--", alpha=0.85)
    ax.axvline(T_star_empirical, color="#2d3748", linestyle=":",
               alpha=0.7, label=f"empirical $T^*$ = {T_star_empirical:.2f}")
    if pred["valid"]:
        ax.axvline(T_star_theory, color="#c53030", linestyle="-",
                   alpha=0.85, label=f"theory $T^*$ = {T_star_theory:.2f}")
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("ECE / bound on ECE")
    ax.set_title(f"Optimal temperature: theory vs empirical\n"
                 f"(K={K}, $\\rho^2$={rho2:.2f}, $a$={a_used:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "T_star_curve.pdf"))
    fig.savefig(os.path.join(args.out, "T_star_curve.png"), dpi=150)
    plt.close(fig)

    # Plot 2: Decomposition of bound (bias term + variance term) vs T
    var_term = bounds - biases
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(T_arr, biases, label="bias term", color="#c53030", linewidth=1.8)
    ax.plot(T_arr, var_term, label=r"variance term $C(T)\sqrt{\mathrm{FNC1}}$",
            color="#2b6cb0", linewidth=1.8)
    ax.plot(T_arr, bounds, label="bound (sum)", color="#2d3748",
            linewidth=2.2, linestyle="--", alpha=0.9)
    if pred["valid"]:
        ax.axvline(T_star_theory, color="#c53030", linestyle=":", alpha=0.7,
                   label=f"theory $T^*$ = {T_star_theory:.2f}")
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("bound terms")
    ax.set_title("Bias-variance trade-off in temperature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "T_decomposition.pdf"))
    fig.savefig(os.path.join(args.out, "T_decomposition.png"), dpi=150)
    plt.close(fig)

    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
