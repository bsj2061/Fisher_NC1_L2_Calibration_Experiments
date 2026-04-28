"""
Generate synthetic snapshots that mimic NC dynamics, then run exp4_trajectory.py
on them to validate the analysis pipeline without needing real training.

The synthetic dynamics:
  - At epoch 0, mu_c is random near origin, W is random.
  - Over epochs, mu_c migrates toward a simplex ETF configuration in R^d
    (we pick the target ETF in advance), and W migrates toward W = M^T
    (the NC3 self-duality target).
  - We add a controllable amount of within-class spread that decays over
    epochs (simulating NC1 progress).

This setup gives us *known ground truth* for trajectory analysis: the
m-coord trajectory should move from barycenter to vertices, the e-coord
trajectory should expand outward into ETF, and the Pythagorean residual
should be small at all times (since we're working in the exact softmax
exponential family).
"""
import env_setup  # noqa: F401  -- OpenMP workaround, must be first

import math
import os
import torch
import shutil

from trajectory import simplex_etf_reference


def make_synthetic_snapshots(
    K: int = 10,
    d: int = 64,
    n_epochs: int = 30,
    rho_final: float = 4.0,
    diag_per_class: int = 30,
    noise_init: float = 0.5,
    noise_final: float = 0.1,
    out_dir: str = "synth_snapshots",
    seed: int = 0,
):
    """
    Generate a sequence of NC-like snapshots and save them in the format
    train_with_snapshots.py produces.
    """
    torch.manual_seed(seed)

    # Random orthogonal embedding from R^K to R^d for placing the ETF
    Q = torch.linalg.qr(torch.randn(d, K))[0]   # (d, K)

    # Targets
    target_etf = simplex_etf_reference(K, rho_final)   # (K, K)
    target_mu = (Q @ target_etf.T).T                    # (K, d)
    target_W = target_mu.clone()                        # NC3: W = M^T (transpose convention)

    # Initial state (random, small)
    init_mu = 0.3 * torch.randn(K, d)
    init_W = 0.3 * torch.randn(K, d)

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(n_epochs + 1):
        # Sigmoid-like progress so that NC accelerates mid-training
        # (matches typical NC dynamics: slow start, fast middle, plateau)
        x = (epoch / n_epochs - 0.5) * 8.0
        progress = 1.0 / (1.0 + math.exp(-x))

        mu = (1 - progress) * init_mu + progress * target_mu
        W = (1 - progress) * init_W + progress * target_W

        # Within-class spread decays over time
        sigma = noise_init * (1 - progress) + noise_final * progress

        # Diagnostic batch: per-class samples scattered around mu_c with current sigma
        diag_feats = []
        diag_labs = []
        for c in range(K):
            samples = mu[c] + sigma * torch.randn(diag_per_class, d)
            diag_feats.append(samples)
            diag_labs.append(torch.full((diag_per_class,), c, dtype=torch.long))
        diag_feats = torch.cat(diag_feats)
        diag_labs = torch.cat(diag_labs)

        snap = {
            "epoch": epoch,
            "mu": mu,
            "W": W,
            "diag_features": diag_feats,
            "diag_labels": diag_labs,
        }
        torch.save(snap, os.path.join(out_dir, f"epoch_{epoch:02d}.pt"))

    print(f"Generated {n_epochs + 1} synthetic snapshots in {out_dir}")
    print(f"  K = {K}, d = {d}")
    print(f"  rho: 0 (init) -> {rho_final} (final)")
    print(f"  within-class sigma: {noise_init} -> {noise_final}")


if __name__ == "__main__":
    make_synthetic_snapshots(
        K=10, d=64, n_epochs=30,
        rho_final=4.0,
        diag_per_class=30,
        noise_init=0.5, noise_final=0.1,
        out_dir="synth_snapshots",
        seed=0,
    )
