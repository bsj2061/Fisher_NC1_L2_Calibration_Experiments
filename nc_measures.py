"""
Neural Collapse measures: Euclidean variance, Fisher-NC1, and the
row-space / null-space decomposition that distinguishes them.

All quantities here operate on already-extracted features (h_i, y_i) pairs.
We provide a single FeatureBank class that holds (features, labels) and
exposes all measures as methods on it.

Conventions:
- features tensor shape: (N, d)
- labels tensor shape: (N,) with values in {0, ..., K-1}
- W tensor shape: (K, d)  -- matches nn.Linear convention
"""

from __future__ import annotations
import torch
from dataclasses import dataclass


# ----------------------------------------------------------------------
# Core utilities
# ----------------------------------------------------------------------

def class_means(features: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    """Return (K, d) tensor of class means."""
    d = features.shape[1]
    mu = torch.zeros(K, d, device=features.device, dtype=features.dtype)
    for c in range(K):
        mask = labels == c
        if mask.any():
            mu[c] = features[mask].mean(dim=0)
    return mu


def within_class_covariance(
    features: torch.Tensor,
    labels: torch.Tensor,
    mu: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """
    Pooled within-class covariance Sigma_W = (1/N) sum_i (h_i - mu_{y_i})(h_i - mu_{y_i})^T.
    Shape (d, d).
    """
    d = features.shape[1]
    centered = features - mu[labels]  # (N, d)
    return (centered.T @ centered) / features.shape[0]


def between_class_covariance(mu: torch.Tensor) -> torch.Tensor:
    """
    Between-class covariance Sigma_B = (1/K) sum_c (mu_c - mu_bar)(mu_c - mu_bar)^T.
    Shape (d, d).
    """
    K = mu.shape[0]
    mu_bar = mu.mean(dim=0, keepdim=True)
    delta = mu - mu_bar
    return (delta.T @ delta) / K


# ----------------------------------------------------------------------
# Row-space / Null-space decomposition (the core of the L2-vs-Fisher claim)
# ----------------------------------------------------------------------

@dataclass
class WSubspace:
    """SVD-derived projection onto row space and null space of W."""
    P_row: torch.Tensor   # (d, d) projection onto row(W^T) = output-relevant directions
    P_null: torch.Tensor  # (d, d) projection onto null(W) = output-invariant directions
    rank: int
    d: int

    @property
    def null_dim(self) -> int:
        return self.d - self.rank


def w_subspace(W: torch.Tensor, tol: float | None = None) -> WSubspace:
    """
    Decompose R^d into row(W^T) (+) null(W).
    W has shape (K, d). Row space of W^T = column space of W^T = span of rows of W.

    Use SVD: W = U S V^T  (U: K x K, S: min(K,d), V: d x d truncated).
    Row space basis = first rank(W) columns of V.
    Null space basis = remaining columns of V.
    """
    K, d = W.shape
    # SVD: W = U @ diag(S) @ Vh  with Vh shape (min(K,d), d) when full_matrices=False
    # We want full V (d x d) to access null space directly.
    U, S, Vh = torch.linalg.svd(W, full_matrices=True)  # Vh: (d, d)
    if tol is None:
        tol = max(K, d) * S[0].item() * torch.finfo(W.dtype).eps
    rank = int((S > tol).sum().item())

    V = Vh.T  # columns are right-singular vectors
    V_row = V[:, :rank]            # (d, rank)
    V_null = V[:, rank:]           # (d, d-rank)

    P_row = V_row @ V_row.T        # projection onto row(W^T)
    P_null = V_null @ V_null.T     # projection onto null(W)
    return WSubspace(P_row=P_row, P_null=P_null, rank=rank, d=d)


def variance_decomposition(
    features: torch.Tensor,
    labels: torch.Tensor,
    mu: torch.Tensor,
    sub: WSubspace,
) -> dict:
    """
    Decompose within-class variance into row-space and null-space components.

    Var(l)         = E[||h - mu||^2]
    Var_row(l)     = E[||P_row (h - mu)||^2]   <- calibration-relevant
    Var_null(l)    = E[||P_null (h - mu)||^2]  <- calibration-irrelevant

    By Pythagoras: Var = Var_row + Var_null.
    """
    centered = features - mu[labels]  # (N, d)
    var_total = (centered ** 2).sum(dim=1).mean().item()
    centered_row = centered @ sub.P_row.T
    centered_null = centered @ sub.P_null.T
    var_row = (centered_row ** 2).sum(dim=1).mean().item()
    var_null = (centered_null ** 2).sum(dim=1).mean().item()
    return {
        "var_total": var_total,
        "var_row": var_row,
        "var_null": var_null,
        "null_fraction": var_null / max(var_total, 1e-12),
    }


# ----------------------------------------------------------------------
# Fisher-NC1 (defined via KL divergence on output simplex)
# ----------------------------------------------------------------------

def softmax_log_softmax(logits: torch.Tensor, T: float = 1.0):
    """Numerically stable softmax + log-softmax at temperature T."""
    z = logits / T
    log_p = torch.log_softmax(z, dim=-1)
    p = log_p.exp()
    return p, log_p


def kl_categorical(p: torch.Tensor, log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """
    KL(p || q) = sum_k p_k (log p_k - log q_k).
    Inputs:
      p:     (..., K) probabilities
      log_p: (..., K) log probabilities
      log_q: (..., K) log probabilities of reference distribution
    Returns: (...) KL values.
    """
    return (p * (log_p - log_q)).sum(dim=-1)


def fisher_nc1(
    features: torch.Tensor,
    labels: torch.Tensor,
    W: torch.Tensor,
    mu: torch.Tensor,
    T: float = 1.0,
    K: int | None = None,
) -> dict:
    """
    Fisher-NC1(l) := (1/K) sum_c E_{h|c}[ KL( p(y|h) || p(y|mu_c) ) ]

    Computed at temperature T (default 1.0, matches training).
    """
    if K is None:
        K = mu.shape[0]
    logits = features @ W.T          # (N, K)
    logits_mu = mu @ W.T             # (K, K)

    p, log_p = softmax_log_softmax(logits, T=T)
    _, log_q_mu = softmax_log_softmax(logits_mu, T=T)
    log_q = log_q_mu[labels]          # broadcast: each sample uses its class's mu's distribution

    kl = kl_categorical(p, log_p, log_q)  # (N,)

    # Average per-class then over classes (handle imbalance gently)
    fnc1_per_class = torch.zeros(K, device=features.device, dtype=features.dtype)
    counts = torch.zeros(K, device=features.device, dtype=features.dtype)
    for c in range(K):
        mask = labels == c
        if mask.any():
            fnc1_per_class[c] = kl[mask].mean()
            counts[c] = mask.sum()
    fnc1 = fnc1_per_class[counts > 0].mean().item()
    return {
        "fnc1": fnc1,
        "per_class_kl": fnc1_per_class.cpu(),
        "kl_per_sample_mean": kl.mean().item(),
        "kl_per_sample_std": kl.std().item(),
    }


# ----------------------------------------------------------------------
# Euclidean NC1 (the PDF's Eq. 2; for comparison only)
# ----------------------------------------------------------------------

def euclidean_nc1(
    features: torch.Tensor,
    labels: torch.Tensor,
    K: int,
    eps: float = 1e-8,
) -> float:
    """
    NC1 = (1/K) tr(Sigma_W Sigma_B^+).
    Uses pseudoinverse via torch.linalg.pinv.
    """
    mu = class_means(features, labels, K)
    Sigma_W = within_class_covariance(features, labels, mu, K)
    Sigma_B = between_class_covariance(mu)
    Sigma_B_pinv = torch.linalg.pinv(Sigma_B + eps * torch.eye(Sigma_B.shape[0], device=Sigma_B.device))
    return torch.trace(Sigma_W @ Sigma_B_pinv).item() / K


# ----------------------------------------------------------------------
# ETF geometry diagnostics
# ----------------------------------------------------------------------

def etf_diagnostics(mu: torch.Tensor) -> dict:
    """
    Check how close class means are to a simplex ETF.

    Simplex ETF properties:
    - All ||mu_c||^2 equal (equinorm).
    - All <mu_c, mu_c'> = -rho^2 / (K-1) for c != c' (equiangular).

    Returns:
    - rho2:   E[||mu_c||^2] (the squared ETF radius).
    - equinorm_cv:   coefficient of variation of ||mu_c||^2 (lower = closer to ETF).
    - equiangle_cv:  coefficient of variation of <mu_c, mu_c'> for c!=c'.
    - mean_inner_off_diag: average off-diagonal inner product.
    - expected_off_diag:   theoretical -rho^2/(K-1).
    """
    K, d = mu.shape
    mu_bar = mu.mean(dim=0, keepdim=True)
    mu_centered = mu - mu_bar  # ETF analysis is centered

    norms2 = (mu_centered ** 2).sum(dim=1)  # (K,)
    rho2 = norms2.mean().item()
    equinorm_cv = (norms2.std() / (norms2.mean() + 1e-12)).item()

    G = mu_centered @ mu_centered.T   # (K, K) Gram of centered means
    off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=mu.device)
    off_diag_vals = G[off_diag_mask]
    equiangle_cv = (off_diag_vals.std() / (off_diag_vals.mean().abs() + 1e-12)).item()

    return {
        "rho2": rho2,
        "rho": rho2 ** 0.5,
        "equinorm_cv": equinorm_cv,
        "equiangle_cv": equiangle_cv,
        "mean_inner_off_diag": off_diag_vals.mean().item(),
        "expected_off_diag": -rho2 / (K - 1),
    }


# ----------------------------------------------------------------------
# Wrapper: extract everything in one pass
# ----------------------------------------------------------------------

def all_nc_measures(
    features: torch.Tensor,
    labels: torch.Tensor,
    W: torch.Tensor,
    T: float = 1.0,
) -> dict:
    """One-stop computation of every NC measure used in the paper."""
    K = int(labels.max().item() + 1)
    mu = class_means(features, labels, K)
    sub = w_subspace(W)
    var_decomp = variance_decomposition(features, labels, mu, sub)
    fnc = fisher_nc1(features, labels, W, mu, T=T, K=K)
    nc1_eucl = euclidean_nc1(features, labels, K)
    etf = etf_diagnostics(mu)

    W_op = torch.linalg.matrix_norm(W, ord=2).item()  # spectral norm
    return {
        "K": K,
        "d": features.shape[1],
        "rank_W": sub.rank,
        "null_dim": sub.null_dim,
        "T": T,
        "W_op_norm": W_op,
        "var_total": var_decomp["var_total"],
        "var_row": var_decomp["var_row"],
        "var_null": var_decomp["var_null"],
        "null_fraction": var_decomp["null_fraction"],
        "fnc1": fnc["fnc1"],
        "euclidean_nc1": nc1_eucl,
        "rho2": etf["rho2"],
        "rho": etf["rho"],
        "equinorm_cv": etf["equinorm_cv"],
        "equiangle_cv": etf["equiangle_cv"],
    }
