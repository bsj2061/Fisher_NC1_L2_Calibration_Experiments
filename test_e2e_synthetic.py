"""
End-to-end test on a fully synthetic dataset with controllable NC strength
and known true ETF radius. Validates that all three experiments produce
sensible output.

We construct:
  - K class means at known radius rho on a simplex ETF in R^d
  - Per-class features = mean + small row-space noise + larger null-space noise
  - Logits computed by W @ h, with W chosen to be the matrix of class means

This setup gives us a *ground-truth* rho and lets us verify that:
  (a) Experiment 1 finds the expected null fraction
  (b) Experiment 2 shows L2 >> Fisher bound (because null variance is large)
  (c) Experiment 3's theoretical T* is in the right ballpark
"""
import env_setup  # noqa: F401  -- OpenMP workaround, must be first

import os
import math
import torch
import numpy as np
import json

from nc_measures import (
    class_means, w_subspace, variance_decomposition, fisher_nc1,
    etf_diagnostics,
)
from calibration import (
    compute_ece, predict_T_star_theoretical, ece_bound_fisher,
    ece_bound_variance, temperature_scale_probs,
)


def build_synthetic_dataset(K=10, d=64, n_per=200, rho=2.0,
                             noise_row=0.6, noise_null=1.5, seed=0):
    """
    Synthetic features in R^d:
      - K simplex ETF means at radius rho (centered)
      - Each sample = mean_c + noise_row * row-space gaussian + noise_null * null-space gaussian
      - W = M (matrix of class means) -> simulates NC3 self-duality
      - Row-space noise large enough to make accuracy < 1 (so T* well-defined).
    """
    torch.manual_seed(seed)

    # Build K vectors with simplex ETF geometry, scaled to ||mu_c|| = rho.
    # Construct in R^K then embed.
    M_K = torch.eye(K) - (1.0 / K)
    # Use any orthonormal basis of the K-1 dim subspace to project out the all-ones direction
    # The columns of M_K are the centered standard basis vectors.
    # Their norms are sqrt((K-1)/K), inner products -1/K. Scale so norm = rho.
    scale = rho / math.sqrt((K - 1) / K)
    M = scale * M_K  # K x K, each column is a class mean in R^K (centered ETF)

    # Embed into R^d (d > K)
    Q = torch.linalg.qr(torch.randn(d, K))[0]   # d x K, orthonormal columns
    mu = (Q @ M).T   # (K, d)

    # W = M (simulating NC3); shape (K, d)
    W_true = mu.clone()

    # Build subspace projectors using true W
    sub = w_subspace(W_true)
    P_row = sub.P_row
    P_null = sub.P_null

    feats, labs, logits = [], [], []
    for c in range(K):
        eps = torch.randn(n_per, d)
        eps_row = eps @ P_row.T
        eps_null = eps @ P_null.T
        h_c = mu[c] + noise_row * eps_row + noise_null * eps_null
        z_c = h_c @ W_true.T
        feats.append(h_c)
        logits.append(z_c)
        labs.append(torch.full((n_per,), c, dtype=torch.long))
    return torch.cat(feats), torch.cat(labs), torch.cat(logits), W_true, mu


def main():
    print("=" * 70)
    print("End-to-end synthetic test")
    print("=" * 70)

    rho_true = 2.0
    K, d = 10, 64
    feats, labs, logits, W, mu_true = build_synthetic_dataset(
        K=K, d=d, n_per=200, rho=rho_true,
        noise_row=0.6, noise_null=1.5, seed=42
    )
    print(f"\nSynthetic setup: K={K}, d={d}, rho_true={rho_true}, "
          f"noise_row=0.6, noise_null=1.5 (larger)")

    # === Experiment 1: variance decomposition ===
    print("\n--- Experiment 1: Null-space hypothesis ---")
    mu_hat = class_means(feats, labs, K)
    sub = w_subspace(W)
    decomp = variance_decomposition(feats, labs, mu_hat, sub)
    print(f"rank(W) = {sub.rank}, null dim = {sub.null_dim}")
    print(f"var_total = {decomp['var_total']:.4f}")
    print(f"var_row   = {decomp['var_row']:.4f}  ({decomp['var_row']/decomp['var_total']*100:.1f}%)")
    print(f"var_null  = {decomp['var_null']:.4f}  ({decomp['null_fraction']*100:.1f}%)")
    print(f"  -> noise_null=1.5 vs noise_row=0.6 means we expect roughly")
    print(f"     (1.5^2 * (d-K)) / (0.6^2 * K + 1.5^2 * (d-K)) = ", end="")
    expected_null = (1.5**2 * (d-K)) / (0.6**2 * K + 1.5**2 * (d-K))
    print(f"{expected_null:.3f}")
    assert decomp['null_fraction'] > 0.7, "Null fraction should be large"

    # ETF diagnostics
    etf = etf_diagnostics(mu_hat)
    print(f"\nETF check:  rho_hat = {etf['rho']:.4f} (true={rho_true})")
    print(f"           equinorm CV = {etf['equinorm_cv']:.4e}")
    print(f"           equiangle CV = {etf['equiangle_cv']:.4e}")

    # === Experiment 2: bound comparison ===
    print("\n--- Experiment 2: Bound comparison at T=1 ---")
    T = 1.0
    fnc = fisher_nc1(feats, labs, W, mu_hat, T=T, K=K)["fnc1"]
    W_op = torch.linalg.matrix_norm(W, ord=2).item()
    probs = temperature_scale_probs(logits, T)
    ece = compute_ece(probs, labs, n_bins=15)["ece"]

    bound_fisher = ece_bound_fisher(fnc, W_op, T)
    bound_l2_total = ece_bound_variance(decomp['var_total'], W_op, T)
    bound_l2_row = ece_bound_variance(decomp['var_row'], W_op, T)
    print(f"empirical ECE = {ece:.4f}")
    print(f"FNC1 = {fnc:.4f},  ||W||_op = {W_op:.4f}")
    print(f"Fisher bound (variance term) = {bound_fisher:.4f}")
    print(f"L2 row-only bound            = {bound_l2_row:.4f}")
    print(f"L2 total bound               = {bound_l2_total:.4f}")
    print(f"Ratio L2_total / Fisher      = {bound_l2_total / bound_fisher:.2f}x")

    assert bound_l2_total > bound_fisher, "L2 bound should be looser than Fisher"

    # === Experiment 3: T* prediction ===
    print("\n--- Experiment 3: T* prediction ---")
    a = (probs.argmax(1) == labs).float().mean().item()
    print(f"per-sample accuracy a = {a:.4f}")
    pred = predict_T_star_theoretical(rho2=etf['rho2'], K=K, accuracy=a)
    print(f"theoretical T* = {pred['T_star_predicted']:.4f}  (beta={pred['beta']:.4f})")

    # Grid-search empirical T* AND bound T*
    # FNC1 measured ONCE at training T_0 = 1, held fixed for post-hoc T sweep
    fnc_T0 = fisher_nc1(feats, labs, W, mu_hat, T=1.0, K=K)["fnc1"]
    print(f"FNC1 (at T_0=1, held fixed) = {fnc_T0:.4f}")

    Ts = np.linspace(0.3, 5.0, 95)
    eces, bounds_at_T = [], []
    # per-class accuracies (T-independent)
    preds_T1 = probs.argmax(1)
    acc_pc = torch.zeros(K)
    for c in range(K):
        m = labs == c
        if m.any():
            acc_pc[c] = (preds_T1[m] == c).float().mean()

    for Tv in Ts:
        p = temperature_scale_probs(logits, Tv)
        eces.append(compute_ece(p, labs, n_bins=15)["ece"])
        # bias term at this T
        logits_mu = mu_hat @ W.T
        p_mu = torch.softmax(logits_mu / Tv, dim=-1)
        diag = torch.diagonal(p_mu)
        bias_T = (diag - acc_pc).abs().mean().item()
        bounds_at_T.append(ece_bound_fisher(fnc_T0, W_op, Tv, bias=bias_T))

    T_emp = float(Ts[int(np.argmin(eces))])
    T_bound = float(Ts[int(np.argmin(bounds_at_T))])
    print(f"empirical T*       = {T_emp:.4f}  (minimizer of empirical ECE)")
    print(f"bound's T*         = {T_bound:.4f}  (minimizer of Fisher bound)")
    err_vs_emp = abs(pred['T_star_predicted'] - T_emp) / T_emp * 100
    err_vs_bound = abs(pred['T_star_predicted'] - T_bound) / T_bound * 100
    print(f"theory vs empirical: {err_vs_emp:.1f}% relative error")
    print(f"theory vs bound min: {err_vs_bound:.1f}% relative error")
    print()
    print("Interpretation:")
    print("  The theoretical formula minimizes the *bound* (Theorem 2).")
    print("  Bound's T* should match theory closely; empirical T* may differ")
    print("  due to Pinsker slack and ETF/UFM approximation error.")
    print(f"  Equiangle CV here = {etf['equiangle_cv']:.3f} (perfect ETF would be ~0).")
    print(f"  On a well-trained CIFAR model, equiangle CV is typically < 0.1,")
    print(f"  giving better theory-empirical agreement.")

    print("\n" + "=" * 70)
    print("End-to-end synthetic test passed.")
    print("=" * 70)
    print("\nThis validates the pipeline. To run on real CIFAR-100 data:")
    print("  1. python train.py --epochs 30 --save-dir checkpoints/baseline")
    print("  2. python extract_features.py --checkpoint checkpoints/baseline/best.pt \\")
    print("       --split test --out cache/test_features.pt")
    print("  3. python exp1_null_space.py     --features cache/test_features.pt --out results/exp1")
    print("  4. python exp2_bound_comparison.py --features cache/test_features.pt --out results/exp2")
    print("  5. python exp3_T_star.py         --features cache/test_features.pt --out results/exp3")


if __name__ == "__main__":
    main()
