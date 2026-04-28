"""
Sanity tests for trajectory.py functions.
"""
import env_setup  # noqa: F401  -- OpenMP workaround, must be first

import math
import torch
import os
import shutil
import tempfile

from trajectory import (
    Snapshot, simplex_etf_reference, etf_distance,
    soft_onehot, kl_categorical_pq, pythagorean_decomposition,
    centered_class_logits, class_softmax,
)


def test_etf_reference_is_etf():
    """An exact simplex ETF reference should have all etf_distance metrics ~0."""
    K = 8
    rho = 3.0
    z = simplex_etf_reference(K, rho)
    info = etf_distance(z)
    print(f"[test_etf_ref] rho_emp={info['rho']:.4f} (target {rho}), "
          f"equinorm_cv={info['equinorm_cv']:.2e}, equiangle_cv={info['equiangle_cv']:.2e}, "
          f"procrustes_l2={info['procrustes_l2']:.2e}")
    assert abs(info['rho'] - rho) < 1e-4
    assert info['equinorm_cv'] < 1e-5
    assert info['equiangle_cv'] < 1e-3
    assert info['procrustes_l2'] < 1e-3


def test_kl_zero_at_self():
    """KL(p || p) = 0."""
    p = torch.tensor([[0.1, 0.3, 0.6], [0.4, 0.5, 0.1]])
    kl = kl_categorical_pq(p, p)
    print(f"[test_kl_self] KL(p||p) = {kl.tolist()}")
    assert torch.all(kl.abs() < 1e-6)


def test_kl_positive():
    """KL is non-negative for any p, q."""
    torch.manual_seed(0)
    K = 5
    logits_p = torch.randn(20, K)
    logits_q = torch.randn(20, K)
    p = torch.softmax(logits_p, dim=-1)
    q = torch.softmax(logits_q, dim=-1)
    kl = kl_categorical_pq(p, q)
    print(f"[test_kl_pos] min KL = {kl.min().item():.4f}")
    assert kl.min() >= -1e-6


def test_soft_onehot_sums_to_one():
    K = 7
    e = soft_onehot(K, eps=0.05)
    s = e.sum(dim=-1)
    print(f"[test_soft_onehot] row sums = {s.min().item():.6f} to {s.max().item():.6f}")
    assert torch.all((s - 1).abs() < 1e-6)


def test_pythagorean_at_collapse():
    """
    When features have collapsed exactly to class means (within-class spread = 0),
    KL(p_i || q_{y_i}) = 0, so KL(p || e) should equal KL(q || e) exactly.
    Pythagorean residual should be ~0.
    """
    torch.manual_seed(1)
    K, d = 5, 16
    mu = torch.randn(K, d)
    W = torch.randn(K, d) * 0.5
    n_per = 30
    # Diag features: exactly the class means (no spread)
    diag_feats = mu.repeat_interleave(n_per, dim=0)
    diag_labs = torch.arange(K).repeat_interleave(n_per)
    snap = Snapshot(epoch=0, mu=mu, W=W,
                    diag_features=diag_feats, diag_labels=diag_labs)
    pyth = pythagorean_decomposition(snap, T=1.0, soft_eps=1e-3)
    print(f"[test_pyth_collapse] KL(p||q)={pyth['mean_KL_p_to_q']:.2e}, "
          f"residual_rel={pyth['mean_residual_rel']:.2e}")
    assert pyth['mean_KL_p_to_q'] < 1e-8, "KL(p||q) should be 0 at collapse"
    assert pyth['mean_residual_rel'] < 1e-4, "Pythagorean should hold exactly"


def test_softmax_invariance():
    """class_softmax(snap) should equal softmax of class_logits(snap)."""
    torch.manual_seed(2)
    K, d = 4, 12
    snap = Snapshot(
        epoch=0,
        mu=torch.randn(K, d),
        W=torch.randn(K, d),
        diag_features=torch.randn(20, d),
        diag_labels=torch.randint(0, K, (20,)),
    )
    q = class_softmax(snap, T=1.0)
    z = snap.mu @ snap.W.T
    q_check = torch.softmax(z, dim=-1)
    err = (q - q_check).abs().max().item()
    print(f"[test_softmax] max diff = {err:.2e}")
    assert err < 1e-6


def test_e_coords_centered():
    """centered_class_logits should have zero column-wise mean."""
    torch.manual_seed(3)
    K, d = 6, 20
    snap = Snapshot(
        epoch=0,
        mu=torch.randn(K, d),
        W=torch.randn(K, d),
        diag_features=torch.randn(10, d),
        diag_labels=torch.randint(0, K, (10,)),
    )
    z = centered_class_logits(snap)
    col_means = z.mean(dim=0)
    print(f"[test_e_centered] max abs col mean = {col_means.abs().max().item():.2e}")
    assert col_means.abs().max() < 1e-5


def test_load_save_snapshots():
    """Round-trip save and load of a snapshot dict."""
    from trajectory import load_snapshots
    K, d = 4, 8
    tmp = tempfile.mkdtemp()
    try:
        for ep in [0, 5, 10]:
            torch.save({
                "epoch": ep,
                "mu": torch.randn(K, d) * (1 + ep * 0.1),
                "W": torch.randn(K, d) * 0.5,
                "diag_features": torch.randn(20, d),
                "diag_labels": torch.randint(0, K, (20,)),
            }, os.path.join(tmp, f"epoch_{ep:02d}.pt"))
        snaps = load_snapshots(tmp)
        epochs = [s.epoch for s in snaps]
        print(f"[test_loadsave] epochs = {epochs}")
        assert epochs == [0, 5, 10]
        assert snaps[0].K == K and snaps[0].d == d
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    print("Running trajectory sanity tests...\n")
    test_etf_reference_is_etf()
    test_kl_zero_at_self()
    test_kl_positive()
    test_soft_onehot_sums_to_one()
    test_pythagorean_at_collapse()
    test_softmax_invariance()
    test_e_coords_centered()
    test_load_save_snapshots()
    print("\nAll trajectory sanity tests passed.")
