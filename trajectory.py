"""
Trajectory analysis in dual coordinates.

Given snapshots (mu, W, diag_features, diag_labels) collected over training,
extract:

  m-coordinates: q_c(t) := softmax(W(t) mu_c(t))    in probability simplex
  e-coordinates: z_c(t) := W(t) mu_c(t)             in centered logit space

and validate the Pythagorean decomposition

    KL(p_i || e_c) = KL(p_i || q_c) + KL(q_c || e_c)

(approximately, since e_c = one-hot is a boundary point of the simplex
 and KL there is technically infinite -- we use a 'soft one-hot' with
 small epsilon mass on off-diagonals as a regularization).

Functions here are pure: they take loaded snapshots and return numbers/arrays.
The plotting and JSON emission live in exp4_trajectory.py.
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch


# ----------------------------------------------------------------------
# Snapshot loading
# ----------------------------------------------------------------------

@dataclass
class Snapshot:
    epoch: int
    mu: torch.Tensor          # (K, d)
    W: torch.Tensor           # (K, d)
    diag_features: torch.Tensor  # (N_d, d)
    diag_labels: torch.Tensor    # (N_d,)

    @property
    def K(self):
        return self.mu.shape[0]

    @property
    def d(self):
        return self.mu.shape[1]


def load_snapshots(snap_dir: str) -> List[Snapshot]:
    """Load all epoch_XX.pt files in chronological order."""
    files = [f for f in os.listdir(snap_dir) if f.startswith("epoch_") and f.endswith(".pt")]
    files = sorted(files, key=lambda f: int(f.split("_")[1].split(".")[0]))
    snaps = []
    for f in files:
        d = torch.load(os.path.join(snap_dir, f), map_location="cpu", weights_only=False)
        snaps.append(Snapshot(
            epoch=d["epoch"], mu=d["mu"], W=d["W"],
            diag_features=d["diag_features"], diag_labels=d["diag_labels"],
        ))
    return snaps


# ----------------------------------------------------------------------
# Coordinate extraction
# ----------------------------------------------------------------------

def class_logits(snap: Snapshot) -> torch.Tensor:
    """E-coordinates: z_c = W mu_c, shape (K, K)."""
    return snap.mu @ snap.W.T


def class_softmax(snap: Snapshot, T: float = 1.0) -> torch.Tensor:
    """M-coordinates: q_c = softmax(W mu_c / T), shape (K, K)."""
    return torch.softmax(class_logits(snap) / T, dim=-1)


def centered_class_logits(snap: Snapshot) -> torch.Tensor:
    """
    Centered e-coords: z_c - mean_c z_c, so the K points live in R^{K-1}
    (orthogonal to the all-ones direction). This is the standard parametrization
    that removes the categorical distribution's redundant degree of freedom.
    """
    z = class_logits(snap)
    return z - z.mean(dim=0, keepdim=True)


# ----------------------------------------------------------------------
# Simplex ETF reference and distance from it
# ----------------------------------------------------------------------

def simplex_etf_reference(K: int, radius: float) -> torch.Tensor:
    """
    Construct K vectors in R^K that form a centered simplex ETF of given radius.
    Each vector is sqrt(K/(K-1)) * (e_c - 1/K), then scaled to ||v|| = radius.
    Inner products: v_c . v_{c'} = -radius^2 / (K-1) for c != c'.
    """
    base = torch.eye(K) - (1.0 / K) * torch.ones(K, K)   # columns = centered standard basis
    base = base.T  # rows = centered ones, K x K
    norms = base.norm(dim=1, keepdim=True)
    base = base / norms                                   # unit row vectors
    return radius * base


def etf_distance(z_centered: torch.Tensor) -> dict:
    """
    Measure how close K centered logit vectors are to a simplex ETF.

    We compute:
      - rho:           empirical mean radius
      - equinorm_cv:   coefficient of variation of |z_c|
      - equiangle_cv:  coefficient of variation of <z_c, z_{c'}> for c != c'
      - mean_inner:    mean of off-diagonal inner products
      - expected:      theoretical -rho^2 / (K-1)
      - permuted_l2:   minimum L2 distance to any signed/permuted ETF reference
                       (this is rotation-and-permutation-invariant)

    The 'permuted_l2' is approximate (we just align via a least-squares
    orthogonal Procrustes fit), but it captures shape distance well.
    """
    K = z_centered.shape[0]
    norms2 = (z_centered ** 2).sum(dim=1)
    rho2 = norms2.mean().item()
    rho = math.sqrt(rho2)
    equinorm_cv = (norms2.std() / (norms2.mean() + 1e-12)).item()

    G = z_centered @ z_centered.T
    off_mask = ~torch.eye(K, dtype=torch.bool)
    off = G[off_mask]
    mean_inner = off.mean().item()
    equiangle_cv = (off.std() / (off.mean().abs() + 1e-12)).item()
    expected = -rho2 / (K - 1)

    # Procrustes-aligned L2 distance to ETF reference
    ref = simplex_etf_reference(K, rho)
    # Find orthogonal R minimizing ||z - ref R||_F via SVD
    M = ref.T @ z_centered                                # (K, K)
    U, _, Vt = torch.linalg.svd(M)
    R = U @ Vt                                            # (K, K)
    aligned_ref = ref @ R
    procrustes_l2 = (z_centered - aligned_ref).norm().item()

    return {
        "rho": rho, "rho2": rho2,
        "equinorm_cv": equinorm_cv, "equiangle_cv": equiangle_cv,
        "mean_inner": mean_inner, "expected_inner": expected,
        "procrustes_l2": procrustes_l2,
    }


# ----------------------------------------------------------------------
# Pythagorean decomposition
# ----------------------------------------------------------------------

def soft_onehot(K: int, eps: float = 1e-3) -> torch.Tensor:
    """
    'Soft' one-hot vertices of the simplex with mass eps spread on off-diagonal.
    Used so that KL(q || e_c) is finite for q on the simplex interior.
    Returns (K, K).
    """
    return (1 - eps) * torch.eye(K) + (eps / (K - 1)) * (1 - torch.eye(K))


def kl_categorical_pq(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    KL(p || q) for categorical distributions. p, q both shape (..., K).
    Returns (...) tensor.
    """
    log_ratio = torch.log(p.clamp_min(eps)) - torch.log(q.clamp_min(eps))
    return (p * log_ratio).sum(dim=-1)


def pythagorean_decomposition(snap: Snapshot, T: float = 1.0,
                              soft_eps: float = 1e-3) -> dict:
    """
    For diagnostic samples (h_i, y_i), let:
      p_i = softmax(W h_i / T)
      q_c = softmax(W mu_c / T)
      e_c = soft-onehot at class c

    Compute:
      A_i = KL(p_i || e_{y_i})           total
      B_i = KL(p_i || q_{y_i})           within-class (NC1-flavored)
      C_c = KL(q_c || e_c)               class-level (NC2/calibration-flavored)

    Pythagorean check: in the *exact* dually-flat setting, A_i = B_i + C_{y_i}.
    Empirically we report mean relative residual:
        residual_i = |A_i - B_i - C_{y_i}| / max(A_i, eps)

    Returns dict with mean values and residual statistics.
    """
    K = snap.K
    W = snap.W
    mu = snap.mu
    h = snap.diag_features
    y = snap.diag_labels

    # Output distributions
    p = torch.softmax((h @ W.T) / T, dim=-1)        # (N, K)
    q = torch.softmax((mu @ W.T) / T, dim=-1)       # (K, K)
    e = soft_onehot(K, eps=soft_eps)                # (K, K)

    q_for_each = q[y]                                # (N, K)
    e_for_each = e[y]                                # (N, K)
    e_for_class_c = e                                # (K, K)

    A = kl_categorical_pq(p, e_for_each)            # (N,)
    B = kl_categorical_pq(p, q_for_each)            # (N,)
    C_per_class = kl_categorical_pq(q, e_for_class_c)  # (K,)
    C_for_each = C_per_class[y]                     # (N,)

    pred = B + C_for_each
    residual = (A - pred).abs()
    rel_residual = residual / A.clamp_min(1e-8)

    return {
        "mean_KL_p_to_e": A.mean().item(),       # the "total" term
        "mean_KL_p_to_q": B.mean().item(),       # NC1 / within-class spread (in distribution space)
        "mean_KL_q_to_e": C_per_class.mean().item(),  # NC2 / class-level miscalibration
        "mean_residual_abs": residual.mean().item(),
        "mean_residual_rel": rel_residual.mean().item(),
        "max_residual_rel": rel_residual.max().item(),
        "C_per_class": C_per_class,
    }


# ----------------------------------------------------------------------
# Trajectory in 2D
# ----------------------------------------------------------------------

def project_simplex_to_2d(probs: torch.Tensor, K: int) -> np.ndarray:
    """
    Project K-dimensional probability vectors to 2D for visualization.

    We use a fixed K-1 -> 2 projection: take the top-2 principal directions
    of the K vertices (one-hot vectors) themselves under PCA. This gives a
    canonical 2D layout of the simplex independent of training run.

    For K=3 this is the standard ternary plot.
    For K=10 it gives a meaningful 2D 'shadow' of the simplex.
    """
    vertices = torch.eye(K)
    centered = vertices - vertices.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
    proj = Vt[:2]                              # (2, K)

    P = probs - probs.mean(dim=-1, keepdim=True)
    coords = P @ proj.T                        # (..., 2)
    return coords.cpu().numpy()


def m_coord_trajectory(snaps: List[Snapshot], T: float = 1.0) -> np.ndarray:
    """
    Returns array of shape (n_snapshots, K, K): q_c(t) for each snapshot t.
    """
    return np.stack([class_softmax(s, T=T).cpu().numpy() for s in snaps], axis=0)


def e_coord_trajectory(snaps: List[Snapshot]) -> np.ndarray:
    """
    Returns array of shape (n_snapshots, K, K): centered z_c(t) for each snapshot.
    """
    return np.stack([centered_class_logits(s).cpu().numpy() for s in snaps], axis=0)


# ----------------------------------------------------------------------
# Convergence summary across snapshots
# ----------------------------------------------------------------------

def compute_trajectory_summary(snaps: List[Snapshot], T: float = 1.0) -> dict:
    """
    Per-epoch summary: ETF metrics, mean Pythagorean residual, m-coord progress
    (KL from barycenter to current q_c), etc.
    """
    K = snaps[0].K
    barycenter = torch.full((K,), 1.0 / K)
    rows = []
    for s in snaps:
        z_centered = centered_class_logits(s)
        etf = etf_distance(z_centered)
        pyth = pythagorean_decomposition(s, T=T)
        q = class_softmax(s, T=T)                    # (K, K)
        # KL(q_c || barycenter) -- reverse, so it's vertex-ward progress
        kl_to_bary = kl_categorical_pq(q, barycenter.expand(K, K)).mean().item()
        # KL(q_c || e_c) using soft one-hot
        kl_q_to_vertex = pyth["mean_KL_q_to_e"]

        rows.append({
            "epoch": s.epoch,
            "rho": etf["rho"],
            "equinorm_cv": etf["equinorm_cv"],
            "equiangle_cv": etf["equiangle_cv"],
            "etf_procrustes_l2": etf["procrustes_l2"],
            "kl_q_to_barycenter": kl_to_bary,
            "kl_q_to_vertex": kl_q_to_vertex,
            "kl_p_to_q": pyth["mean_KL_p_to_q"],
            "kl_p_to_e": pyth["mean_KL_p_to_e"],
            "pythagorean_residual_rel": pyth["mean_residual_rel"],
            "pythagorean_residual_max": pyth["max_residual_rel"],
        })
    return {"rows": rows}
