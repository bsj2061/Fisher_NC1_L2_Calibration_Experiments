"""
Calibration utilities:
- ECE computation (binned).
- Grid search for optimal temperature T*.
- Theoretical T* prediction from the Fisher framework.
- Bound comparison: L2-based vs Fisher-based bound on ECE.
"""

from __future__ import annotations
import math
import torch
from typing import Optional


# ----------------------------------------------------------------------
# Expected Calibration Error
# ----------------------------------------------------------------------

def compute_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> dict:
    """
    Standard ECE with equal-width bins on max-confidence.

    probs: (N, K) probability vectors after softmax.
    labels: (N,) integer class labels.
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == labels).float()

    bin_edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)
    bin_stats = []
    N = probs.shape[0]

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.any():
            acc_bin = accuracies[mask].mean()
            conf_bin = confidences[mask].mean()
            weight = mask.float().mean()
            ece = ece + weight * (acc_bin - conf_bin).abs()
            bin_stats.append({
                "bin": i,
                "lo": lo.item(), "hi": hi.item(),
                "count": int(mask.sum().item()),
                "acc": acc_bin.item(),
                "conf": conf_bin.item(),
            })
    return {"ece": ece.item(), "bin_stats": bin_stats}


def temperature_scale_probs(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Apply temperature T to logits and softmax."""
    return torch.softmax(logits / T, dim=-1)


# ----------------------------------------------------------------------
# Grid search for optimal T (the standard validation-set baseline)
# ----------------------------------------------------------------------

def grid_search_T(
    logits: torch.Tensor,
    labels: torch.Tensor,
    T_grid: Optional[torch.Tensor] = None,
    n_bins: int = 15,
) -> dict:
    """
    Find T* that minimizes ECE on the given (logits, labels) set.
    """
    if T_grid is None:
        T_grid = torch.cat([
            torch.linspace(0.1, 1.0, 19),
            torch.linspace(1.0, 5.0, 41),
        ])

    eces = []
    for T in T_grid:
        probs = temperature_scale_probs(logits, T.item())
        e = compute_ece(probs, labels, n_bins=n_bins)["ece"]
        eces.append(e)
    eces_t = torch.tensor(eces)
    idx_star = int(eces_t.argmin().item())
    return {
        "T_grid": T_grid.tolist(),
        "ece_grid": eces,
        "T_star": T_grid[idx_star].item(),
        "ece_at_T_star": eces[idx_star],
    }


# ----------------------------------------------------------------------
# Theoretical T* prediction (the new derivation)
# ----------------------------------------------------------------------

def predict_T_star_theoretical(
    rho2: float,
    K: int,
    accuracy: float,
) -> dict:
    """
    Predict T* from the closed form derived from Theorem 2 minimization:

        T* = beta / log( a (K-1) / (1-a) )
        beta = rho^2 K / (K-1)

    Args:
      rho2:     E[||mu_c - mu_bar||^2], the squared ETF radius (centered class means)
      K:        number of classes
      accuracy: per-class accuracy a in (1/K, 1)

    Returns:
      Dict with T_star_predicted and intermediate quantities.

    Note: This formula assumes ETF + balanced data + the bound's bias term
    vanishes at p_T(y_c|mu_c) = a. Both assumptions are stated in the paper.
    """
    beta = rho2 * K / (K - 1)
    if not (1.0 / K < accuracy < 1.0):
        return {
            "T_star_predicted": float("nan"),
            "beta": beta,
            "log_arg": float("nan"),
            "valid": False,
            "reason": f"accuracy {accuracy} must lie in (1/K, 1) = ({1.0/K:.4f}, 1.0)",
        }
    log_arg = accuracy * (K - 1) / (1 - accuracy)
    T_pred = beta / math.log(log_arg)
    return {
        "T_star_predicted": T_pred,
        "beta": beta,
        "log_arg": log_arg,
        "valid": True,
    }


# ----------------------------------------------------------------------
# Bound comparison: L2 vs Fisher
# ----------------------------------------------------------------------

def ece_bound_fisher(
    fnc1: float,
    W_op_norm: float,
    T: float,
    bias: float = 0.0,
) -> float:
    """
    Theorem 2 form:  ECE <= bias + (||W||_op / T sqrt(2)) sqrt(FNC1)
    """
    return bias + (W_op_norm / (T * math.sqrt(2))) * math.sqrt(max(fnc1, 0.0))


def ece_bound_variance(
    var_total: float,
    W_op_norm: float,
    T: float,
    bias: float = 0.0,
) -> float:
    """
    L2-based bound (corollary of Theorem 1(a) plugged into Pinsker chain):
        FNC1 <= (||W||_op^2 / 8 T^2) * Var
        ECE  <= bias + (||W||_op / T sqrt(2)) sqrt( (||W||_op^2 / 8 T^2) Var )
              = bias + (||W||_op^2 / (4 T^2)) sqrt(Var) / sqrt(2) ... but careful:

    Plugging the upper bound on FNC1 directly into the Fisher ECE bound:
        ECE <= bias + (||W||_op / T sqrt(2)) * sqrt( ||W||_op^2 Var / (8 T^2) )
             = bias + ||W||_op^2 sqrt(Var) / (T^2 * 4)
    """
    return bias + (W_op_norm ** 2 / (4 * T ** 2)) * math.sqrt(max(var_total, 0.0))


def ece_bound_variance_row(
    var_row: float,
    W_op_norm: float,
    T: float,
    bias: float = 0.0,
) -> float:
    """
    A 'fair' L2-based bound that uses only the row-space component of variance.
    This is what an oracle-aware practitioner could compute if they knew
    to project out the null space. It should match the Fisher bound up to
    spectral-norm slack.
    """
    return ece_bound_variance(var_row, W_op_norm, T, bias=bias)
