"""
Sanity tests on synthetic data.

This file validates that:
  1. nc_measures.py functions don't crash on a small ETF-like setup.
  2. variance decomposition satisfies Pythagoras (var_total = var_row + var_null).
  3. predict_T_star_theoretical returns sensible values on toy inputs.
  4. Fisher-NC1 is small when features collapse to class means.
  5. Fisher-NC1 is much smaller than L2 variance when spread is null-space-aligned.

Run:
  python test_sanity.py
"""
import env_setup  # noqa: F401  -- OpenMP workaround, must be first

import math
import torch
import numpy as np

from nc_measures import (
    class_means, w_subspace, variance_decomposition, fisher_nc1,
    euclidean_nc1, etf_diagnostics,
)
from calibration import (
    predict_T_star_theoretical, ece_bound_fisher, ece_bound_variance,
    compute_ece, temperature_scale_probs,
)


def make_etf_means(K, d, scale=1.0, seed=0):
    """Construct K vectors in R^d that approximately form a simplex ETF."""
    torch.manual_seed(seed)
    A = torch.randn(K, K)
    Q, _ = torch.linalg.qr(A)               # K x K orthogonal
    M = torch.eye(K) - (1.0 / K) * torch.ones(K, K)  # centering projector
    # Build K vectors on a simplex in R^K, then embed into R^d
    base = math.sqrt(K / (K - 1)) * (Q @ M)        # K x K, columns sum to 0
    pad = torch.zeros(d - K, K)
    embedded = torch.cat([base, pad], dim=0).T   # (K, d)
    return embedded * scale


def test_pythagoras():
    """variance decomposition should satisfy var_total = var_row + var_null."""
    torch.manual_seed(1)
    K, d, n_per = 5, 16, 50
    mu = torch.randn(K, d) * 2.0
    W = torch.randn(K, d) * 0.5
    feats = []
    labs = []
    for c in range(K):
        feats.append(mu[c] + 0.3 * torch.randn(n_per, d))
        labs.append(torch.full((n_per,), c, dtype=torch.long))
    feats = torch.cat(feats)
    labs = torch.cat(labs)
    mu_hat = class_means(feats, labs, K)
    sub = w_subspace(W)
    decomp = variance_decomposition(feats, labs, mu_hat, sub)
    err = abs(decomp["var_total"] - (decomp["var_row"] + decomp["var_null"]))
    rel = err / decomp["var_total"]
    print(f"[test_pythagoras] var_total={decomp['var_total']:.4f}, "
          f"var_row+var_null={decomp['var_row']+decomp['var_null']:.4f}, "
          f"rel err={rel:.2e}")
    assert rel < 1e-5, f"Pythagoras violated: rel err {rel}"


def test_null_space_invisible_to_fisher():
    """
    Construct features that vary ONLY in W's null space.
    -> L2 variance is large, Fisher NC1 should be ~0.
    """
    torch.manual_seed(2)
    K, d = 5, 32
    # W has rank K, so null space has dimension d-K=27
    W = torch.randn(K, d) * 0.5
    sub = w_subspace(W)
    assert sub.null_dim == d - K, f"Expected null dim {d-K}, got {sub.null_dim}"

    # Place class means in row space
    mu_centers = torch.randn(K, d)
    mu_centers = mu_centers @ sub.P_row.T  # project into row space

    # Add per-sample noise ENTIRELY in null space
    n_per = 100
    feats, labs = [], []
    for c in range(K):
        noise = torch.randn(n_per, d)
        noise_null = noise @ sub.P_null.T   # project noise to null space
        feats.append(mu_centers[c] + 0.7 * noise_null)
        labs.append(torch.full((n_per,), c, dtype=torch.long))
    feats = torch.cat(feats)
    labs = torch.cat(labs)
    mu_hat = class_means(feats, labs, K)
    decomp = variance_decomposition(feats, labs, mu_hat, sub)
    fnc = fisher_nc1(feats, labs, W, mu_hat, T=1.0, K=K)
    print(f"[test_null_space] var_total={decomp['var_total']:.4f}, "
          f"var_row={decomp['var_row']:.4e}, var_null={decomp['var_null']:.4f}")
    print(f"                 FNC1={fnc['fnc1']:.4e}")
    # With noise only in null space, var_row should be near zero (only sample-mean est. error)
    assert decomp["var_row"] < 1e-3, "var_row should be ~0 when noise is null-space-only"
    # FNC1 should be tiny because output distributions are identical for any null-space deviation
    assert fnc["fnc1"] < 1e-6, f"FNC1 should be ~0 for null-space-only noise, got {fnc['fnc1']}"


def test_fisher_collapse_when_features_collapse():
    """All features = class mean -> Fisher NC1 should be exactly 0."""
    torch.manual_seed(3)
    K, d = 4, 16
    W = torch.randn(K, d) * 0.4
    mu = torch.randn(K, d) * 1.5
    n_per = 30
    feats = mu.repeat_interleave(n_per, dim=0)
    labs = torch.arange(K).repeat_interleave(n_per)
    mu_hat = class_means(feats, labs, K)
    fnc = fisher_nc1(feats, labs, W, mu_hat, T=1.0, K=K)
    print(f"[test_collapse] FNC1 when features = means: {fnc['fnc1']:.2e}")
    assert fnc["fnc1"] < 1e-10, f"FNC1 should be 0 for collapsed features, got {fnc['fnc1']}"


def test_T_star_prediction_basic():
    """Sanity-check the T* formula: should be positive and finite for normal inputs."""
    pred = predict_T_star_theoretical(rho2=10.0, K=100, accuracy=0.6)
    print(f"[test_T_star] rho2=10, K=100, a=0.6 -> T* = {pred['T_star_predicted']:.3f}")
    assert pred["valid"]
    assert pred["T_star_predicted"] > 0

    # Edge case: a very close to 1
    pred2 = predict_T_star_theoretical(rho2=10.0, K=100, accuracy=0.99)
    print(f"[test_T_star] rho2=10, K=100, a=0.99 -> T* = {pred2['T_star_predicted']:.3f}")
    # Higher accuracy -> more overconfident -> smaller T* (since less correction needed... actually
    # higher a means log_arg larger, so T* smaller)

    # Edge case: a too small
    pred3 = predict_T_star_theoretical(rho2=10.0, K=100, accuracy=0.005)  # below 1/K
    print(f"[test_T_star] a < 1/K: valid={pred3['valid']}")
    assert not pred3["valid"]


def test_ece_bound_ordering():
    """At a fixed T, the L2-total bound should be >= L2-row bound."""
    fnc = 0.5
    var_total = 4.0
    var_row = 1.0
    W_op = 3.0
    T = 1.0
    b_fisher = ece_bound_fisher(fnc, W_op, T)
    b_l2_total = ece_bound_variance(var_total, W_op, T)
    b_l2_row = ece_bound_variance(var_row, W_op, T)
    print(f"[test_bound_ordering] Fisher={b_fisher:.3f}, L2_row={b_l2_row:.3f}, L2_total={b_l2_total:.3f}")
    assert b_l2_total >= b_l2_row, "L2 total should be >= L2 row"


def test_etf_geometry():
    """An exact simplex ETF should have low equinorm and equiangle CV."""
    K, d = 10, 20
    mu = make_etf_means(K, d, scale=2.0, seed=0)
    info = etf_diagnostics(mu)
    print(f"[test_etf] rho^2={info['rho2']:.4f}, equinorm CV={info['equinorm_cv']:.2e}, "
          f"equiangle CV={info['equiangle_cv']:.2e}")
    assert info["equinorm_cv"] < 1e-4, "ETF should have ~0 equinorm CV"
    assert info["equiangle_cv"] < 1e-2, "ETF should have small equiangle CV"


def test_ece_basic():
    """ECE on perfectly confident & correct predictions should be 0."""
    K, N = 5, 100
    labels = torch.randint(0, K, (N,))
    one_hot = torch.zeros(N, K)
    one_hot[torch.arange(N), labels] = 1.0
    e = compute_ece(one_hot, labels, n_bins=15)["ece"]
    print(f"[test_ece] perfect predictions ECE = {e:.4f}")
    assert e < 0.01

    # Uniform predictions on N items, accuracy=1/K => ECE near |1/K - 1/K| = 0
    uniform = torch.full((N, K), 1.0 / K)
    e2 = compute_ece(uniform, labels, n_bins=15)["ece"]
    print(f"[test_ece] uniform predictions ECE = {e2:.4f}")
    # Acc on uniform predictions is whatever argmax is; not super tight but should be small-ish
    # Just sanity: it should be <= 1.


if __name__ == "__main__":
    print("Running sanity tests...\n")
    test_pythagoras()
    test_null_space_invisible_to_fisher()
    test_fisher_collapse_when_features_collapse()
    test_T_star_prediction_basic()
    test_ece_bound_ordering()
    test_etf_geometry()
    test_ece_basic()
    print("\nAll sanity tests passed.")
