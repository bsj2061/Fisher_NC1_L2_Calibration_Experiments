"""
Training loop that periodically snapshots:
  - classifier weight W
  - class means mu_c (estimated from train features)
  - per-class output distribution p(y | mu_c)
  - a small per-sample diagnostic batch (for Pythagorean decomposition)

Snapshots are saved to <save_dir>/snapshots/epoch_<NN>.pt and let us
reconstruct the trajectory of class-mean output distributions in both
m-coordinates (probability simplex) and e-coordinates (centered logits).

Run:
    python train_with_snapshots.py --epochs 30 --save-dir checkpoints/snap \
                                   --snapshot-every 1

For CIFAR-10 (K=10) the trajectory plots are most readable; for K=100 we
project to 2D first.
"""
import env_setup  # noqa: F401  -- OpenMP workaround, must be first

import argparse
import json
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from models import ResNet18Cifar


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def get_loaders(dataset: str, batch_size: int, data_root: str = "./data", num_workers: int = 4):
    """CIFAR-10 or CIFAR-100 loaders."""
    if dataset == "cifar10":
        DS = torchvision.datasets.CIFAR10
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    elif dataset == "cifar100":
        DS = torchvision.datasets.CIFAR100
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_set = DS(data_root, train=True, download=True, transform=train_tf)
    test_set = DS(data_root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ----------------------------------------------------------------------
# Snapshot machinery
# ----------------------------------------------------------------------

@torch.no_grad()
def collect_class_means(model, loader, K, device, max_batches: int | None = None):
    """
    Estimate class means mu_c by averaging features of training samples per class.
    """
    model.eval()
    d = model.feature_dim
    sums = torch.zeros(K, d, device=device)
    counts = torch.zeros(K, device=device)
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        h = model.features(x)
        for c in range(K):
            m = (y == c)
            if m.any():
                sums[c] += h[m].sum(dim=0)
                counts[c] += m.sum()
    counts = counts.clamp_min(1.0)
    mu = sums / counts.unsqueeze(1)
    return mu  # (K, d)


@torch.no_grad()
def collect_diagnostic_batch(model, loader, K, device, n_per_class: int = 50):
    """
    Collect up to n_per_class samples per class with their features, labels,
    and logits. This small batch lets us evaluate the Pythagorean
    decomposition KL(p_i || e_c) = KL(p_i || q_c) + KL(q_c || e_c)
    at each snapshot without storing the full dataset.
    """
    model.eval()
    feats, labs = [], []
    counts = torch.zeros(K, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        h = model.features(x)
        for c in range(K):
            if counts[c] >= n_per_class:
                continue
            m = (y == c)
            if not m.any():
                continue
            need = n_per_class - counts[c].item()
            sel = h[m][:need]
            feats.append(sel.cpu())
            labs.append(torch.full((sel.shape[0],), c, dtype=torch.long))
            counts[c] += sel.shape[0]
        if (counts >= n_per_class).all():
            break
    return torch.cat(feats), torch.cat(labs)


@torch.no_grad()
def make_snapshot(model, train_loader, K, device,
                  snapshot_max_batches: int | None = None,
                  diag_per_class: int = 50):
    """
    Build a snapshot dict containing everything needed for trajectory analysis.
    """
    mu = collect_class_means(model, train_loader, K, device,
                             max_batches=snapshot_max_batches)
    diag_feats, diag_labs = collect_diagnostic_batch(model, train_loader, K, device,
                                                     n_per_class=diag_per_class)
    return {
        "mu": mu.cpu(),                         # (K, d)
        "W": model.W.detach().cpu(),            # (K, d)
        "diag_features": diag_feats,            # (N_d, d)
        "diag_labels": diag_labs,               # (N_d,)
    }


# ----------------------------------------------------------------------
# Train / eval steps
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return {"acc": correct / total, "loss": loss_sum / total}


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return {"acc": correct / total, "loss": loss_sum / total}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="CIFAR-10 (K=10) is recommended for cleanest trajectory plots.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="checkpoints/snap")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshot-every", type=int, default=1,
                        help="Take a snapshot every N epochs (also at epoch 0).")
    parser.add_argument("--snapshot-max-batches", type=int, default=None,
                        help="Limit batches used to compute class means (None = full train set).")
    parser.add_argument("--diag-per-class", type=int, default=50,
                        help="Diagnostic samples per class stored in each snapshot.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    snap_dir = os.path.join(args.save_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    K = 10 if args.dataset == "cifar10" else 100
    train_loader, test_loader = get_loaders(args.dataset, args.batch_size, args.data_root)
    model = ResNet18Cifar(num_classes=K).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    t0 = time.time()

    # Snapshot at epoch 0 (random init) so trajectories start from initialization
    print("Taking snapshot at epoch 0 (init)...")
    snap = make_snapshot(model, train_loader, K, args.device,
                         args.snapshot_max_batches, args.diag_per_class)
    snap["epoch"] = 0
    torch.save(snap, os.path.join(snap_dir, "epoch_00.pt"))

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, args.device)
        test_stats = evaluate(model, test_loader, args.device)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {train_stats['loss']:.3f} acc {train_stats['acc']:.3f} | "
              f"test loss {test_stats['loss']:.3f} acc {test_stats['acc']:.3f} | "
              f"{elapsed:.0f}s")
        history.append({
            "epoch": epoch,
            "train": train_stats,
            "test": test_stats,
            "lr": scheduler.get_last_lr()[0],
        })

        if epoch % args.snapshot_every == 0 or epoch == args.epochs:
            snap = make_snapshot(model, train_loader, K, args.device,
                                 args.snapshot_max_batches, args.diag_per_class)
            snap["epoch"] = epoch
            torch.save(snap, os.path.join(snap_dir, f"epoch_{epoch:02d}.pt"))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "last.pt"))
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"\nDone. {args.epochs+1} snapshots saved to {snap_dir}")


if __name__ == "__main__":
    main()
