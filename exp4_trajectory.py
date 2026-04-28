"""
Experiment 4: Trajectory of class-mean output distributions in dual coordinates.

Visualizes:
  (1) m-coord trajectory: how q_c(t) = softmax(W mu_c) moves from the simplex
      barycenter to the simplex vertices over training. 2D PCA projection.
  (2) e-coord trajectory: how centered class logits z_c(t) form the simplex
      ETF over training. 2D PCA projection.
  (3) Pythagorean decomposition: mean KL(p_i || e_c) vs the predicted sum
      KL(p_i || q_c) + KL(q_c || e_c) at each epoch, plus relative residual.
  (4) Convergence metrics per epoch: ETF radius rho, equiangle CV,
      Procrustes distance to perfect ETF.

Inputs:
  --snapshots  Path to a directory of epoch_NN.pt files made by
               train_with_snapshots.py.
  --out        Output directory.

Run:
  python exp4_trajectory.py --snapshots checkpoints/snap/snapshots --out results/exp4
"""
import env_setup  # noqa: F401  -- OpenMP workaround, must be first

import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

from trajectory import (
    load_snapshots, class_softmax, centered_class_logits, etf_distance,
    pythagorean_decomposition, m_coord_trajectory, e_coord_trajectory,
    project_simplex_to_2d, compute_trajectory_summary, simplex_etf_reference,
)


def plot_m_coord_trajectory(snaps, out_path, T=1.0):
    """
    2D PCA projection of probability simplex showing how q_c(t) moves from
    the barycenter to the vertices over training.
    """
    K = snaps[0].K
    n_t = len(snaps)
    epochs = [s.epoch for s in snaps]
    cmap = cm.get_cmap("tab10" if K <= 10 else "viridis")

    # Trajectory tensor shape (T, K, K)
    Q = m_coord_trajectory(snaps, T=T)              # numpy array
    Q_flat = Q.reshape(-1, K)                       # (T*K, K)
    coords_flat = project_simplex_to_2d(torch.from_numpy(Q_flat), K)
    coords = coords_flat.reshape(n_t, K, 2)

    # Reference: vertices and barycenter
    vertex_coords = project_simplex_to_2d(torch.eye(K), K)
    bary_coords = project_simplex_to_2d(torch.full((1, K), 1.0 / K), K)[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    # Draw simplex outline (convex hull of vertices)
    hull = vertex_coords[np.argsort(np.arctan2(
        vertex_coords[:, 1] - bary_coords[1],
        vertex_coords[:, 0] - bary_coords[0],
    ))]
    hull_loop = np.vstack([hull, hull[:1]])
    ax.plot(hull_loop[:, 0], hull_loop[:, 1], color="lightgray", linewidth=0.8, zorder=1)

    # Draw trajectory of each class
    for c in range(K):
        color = cmap(c / max(K - 1, 1)) if K > 10 else cmap(c)
        traj = coords[:, c, :]
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.55, linewidth=1.5, zorder=2)
        # Mark start (epoch 0) and end
        ax.scatter([traj[0, 0]], [traj[0, 1]], color=color, marker="o",
                   edgecolor="black", linewidth=0.5, s=40, zorder=3)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], color=color, marker="*",
                   edgecolor="black", linewidth=0.5, s=120, zorder=4)

    # Mark vertices
    for c in range(K):
        ax.annotate(f"$e_{{{c}}}$", vertex_coords[c],
                    fontsize=9, color="black", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray",
                              lw=0.5, alpha=0.9))
    ax.scatter([bary_coords[0]], [bary_coords[1]], color="black", marker="x", s=50,
               zorder=5, label="barycenter")

    ax.set_title(f"m-coord trajectory of $q_c(t) = \\mathrm{{softmax}}(W \\mu_c)$\n"
                 f"(K={K}, epochs {epochs[0]} → {epochs[-1]}, "
                 f"○ start, ★ end)")
    ax.set_xlabel("PC1 of probability simplex")
    ax.set_ylabel("PC2 of probability simplex")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".png", dpi=150)
    plt.close(fig)


def plot_e_coord_trajectory(snaps, out_path):
    """
    2D PCA projection of *centered logit space* showing how class logits
    expand outward into the simplex ETF configuration.
    """
    K = snaps[0].K
    n_t = len(snaps)
    epochs = [s.epoch for s in snaps]
    cmap = cm.get_cmap("tab10" if K <= 10 else "viridis")

    Z = e_coord_trajectory(snaps)                   # (T, K, K)
    Z_flat = Z.reshape(-1, K)
    # PCA on the *final* configuration so the chosen 2D axes are aligned to ETF
    final_z = Z[-1]                                 # (K, K)
    final_z_centered = final_z - final_z.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(final_z_centered, full_matrices=False)
    proj = Vt[:2]                                   # (2, K)
    coords_flat = Z_flat @ proj.T                   # (T*K, 2)
    coords = coords_flat.reshape(n_t, K, 2)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw final ETF reference
    final_xy = coords[-1]
    final_hull = final_xy[np.argsort(np.arctan2(final_xy[:, 1], final_xy[:, 0]))]
    hull_loop = np.vstack([final_hull, final_hull[:1]])
    ax.plot(hull_loop[:, 0], hull_loop[:, 1], color="lightgray",
            linewidth=0.8, zorder=1, linestyle="--",
            label="final ETF outline")

    # Trajectory
    for c in range(K):
        color = cmap(c / max(K - 1, 1)) if K > 10 else cmap(c)
        traj = coords[:, c, :]
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.55, linewidth=1.5, zorder=2)
        ax.scatter([traj[0, 0]], [traj[0, 1]], color=color, marker="o",
                   edgecolor="black", linewidth=0.5, s=40, zorder=3)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], color=color, marker="*",
                   edgecolor="black", linewidth=0.5, s=120, zorder=4)

    ax.scatter([0], [0], color="black", marker="x", s=50, zorder=5, label="origin")
    ax.set_title(f"e-coord trajectory of $z_c(t) = W \\mu_c$ (centered)\n"
                 f"(K={K}, epochs {epochs[0]} → {epochs[-1]}, "
                 f"○ start, ★ end)")
    ax.set_xlabel("PC1 of final logit configuration")
    ax.set_ylabel("PC2 of final logit configuration")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".png", dpi=150)
    plt.close(fig)


def plot_pythagorean(summary_rows, out_path):
    """Plot KL(p||e), KL(p||q), KL(q||e), and residual over epochs."""
    epochs = [r["epoch"] for r in summary_rows]
    kl_pe = [r["kl_p_to_e"] for r in summary_rows]
    kl_pq = [r["kl_p_to_q"] for r in summary_rows]
    kl_qe = [r["kl_q_to_vertex"] for r in summary_rows]
    sums = [a + b for a, b in zip(kl_pq, kl_qe)]
    resid = [r["pythagorean_residual_rel"] for r in summary_rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)

    ax1.plot(epochs, kl_pe, "o-", label="KL(p || e)  (total)", color="#2d3748", linewidth=2)
    ax1.plot(epochs, sums, "s--", label="KL(p || q) + KL(q || e)  (predicted)",
             color="#c53030", alpha=0.8, linewidth=1.5)
    ax1.plot(epochs, kl_pq, ".-", label="KL(p || q)  (NC1-flavored)",
             color="#2b6cb0", alpha=0.7)
    ax1.plot(epochs, kl_qe, ".-", label="KL(q || e)  (calibration-flavored)",
             color="#805ad5", alpha=0.7)
    ax1.set_ylabel("KL divergence")
    ax1.set_title("Pythagorean decomposition over training")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, resid, "o-", color="#c53030", linewidth=2)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("relative residual\n|KL(p||e) − [KL(p||q) + KL(q||e)]| / KL(p||e)")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_title("Pythagorean residual (lower = exact dual-flat structure)")

    fig.tight_layout()
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".png", dpi=150)
    plt.close(fig)


def plot_etf_metrics(summary_rows, out_path):
    """Plot ETF radius, equiangle CV, and Procrustes distance over epochs."""
    epochs = [r["epoch"] for r in summary_rows]
    rho = [r["rho"] for r in summary_rows]
    equiangle = [r["equiangle_cv"] for r in summary_rows]
    proc = [r["etf_procrustes_l2"] for r in summary_rows]
    proc_normalized = [p / max(r, 1e-8) for p, r in zip(proc, rho)]

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7), sharex=True)

    axes[0].plot(epochs, rho, "o-", color="#2b6cb0", linewidth=2)
    axes[0].set_ylabel(r"ETF radius $\rho$")
    axes[0].set_title("Simplex ETF formation over training")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, equiangle, "o-", color="#805ad5", linewidth=2)
    axes[1].set_ylabel("equiangle CV\n(0 = perfect ETF)")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3, which="both")

    axes[2].plot(epochs, proc_normalized, "o-", color="#c53030", linewidth=2)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel(r"Procrustes L2 / $\rho$" + "\n(shape distance to ideal ETF)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=str, required=True,
                        help="Directory of epoch_NN.pt snapshots")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--T", type=float, default=1.0)
    args = parser.parse_args()

    snaps = load_snapshots(args.snapshots)
    if len(snaps) < 2:
        raise RuntimeError(f"Need at least 2 snapshots, got {len(snaps)} in {args.snapshots}")
    print(f"Loaded {len(snaps)} snapshots: epochs "
          f"{[s.epoch for s in snaps]}")
    print(f"K = {snaps[0].K}, d = {snaps[0].d}")

    os.makedirs(args.out, exist_ok=True)

    # Compute summary
    summary = compute_trajectory_summary(snaps, T=args.T)
    rows = summary["rows"]

    # Plots
    plot_m_coord_trajectory(snaps, os.path.join(args.out, "m_trajectory"), T=args.T)
    plot_e_coord_trajectory(snaps, os.path.join(args.out, "e_trajectory"))
    plot_pythagorean(rows, os.path.join(args.out, "pythagorean"))
    plot_etf_metrics(rows, os.path.join(args.out, "etf_metrics"))

    # Save summary JSON
    with open(os.path.join(args.out, "exp4_summary.json"), "w") as f:
        json.dump({
            "T": args.T,
            "n_snapshots": len(snaps),
            "K": snaps[0].K,
            "d": snaps[0].d,
            "rows": rows,
        }, f, indent=2)

    # Print headline numbers
    first, last = rows[0], rows[-1]
    print(f"\n=== Trajectory summary ===")
    print(f"epoch {first['epoch']:>3d}: rho={first['rho']:.3f}, "
          f"equiangle_cv={first['equiangle_cv']:.3f}, "
          f"KL(p||q)={first['kl_p_to_q']:.3f}, KL(q||e)={first['kl_q_to_vertex']:.3f}")
    print(f"epoch {last['epoch']:>3d}: rho={last['rho']:.3f}, "
          f"equiangle_cv={last['equiangle_cv']:.3f}, "
          f"KL(p||q)={last['kl_p_to_q']:.3f}, KL(q||e)={last['kl_q_to_vertex']:.3f}")
    print(f"Pythagorean residual (final): "
          f"mean rel = {last['pythagorean_residual_rel']:.4f}, "
          f"max rel = {last['pythagorean_residual_max']:.4f}")
    print(f"\nFigures and summary saved to {args.out}")


if __name__ == "__main__":
    main()
