"""
Extract penultimate features and logits from a trained model and cache them.

Usage:
    python extract_features.py --checkpoint checkpoints/baseline/best.pt \
                               --split test --out cache/test_features.pt

Caching is essential because all NC measures and bound computations operate
on (features, labels, logits) triples, and we want to avoid re-running the
forward pass for every experiment.
"""
import env_setup  # noqa: F401  -- OpenMP workaround, must be first

from __future__ import annotations
import argparse
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from models import get_model


def get_loader(split: str, batch_size: int = 256, data_root: str = "./data", num_workers: int = 4):
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    train = (split == "train")
    ds = torchvision.datasets.CIFAR100(data_root, train=train, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    feats, lbls, lgts = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits, h = model(x, return_features=True)
        feats.append(h.cpu())
        lgts.append(logits.cpu())
        lbls.append(y)
    return torch.cat(feats), torch.cat(lbls), torch.cat(lgts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = get_model(num_classes=args.num_classes, device=args.device)
    sd = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(sd)

    loader = get_loader(args.split, data_root=args.data_root)
    feats, lbls, lgts = extract(model, loader, args.device)

    # Save W along with features so all experiments are self-contained.
    W = model.W.cpu()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "features": feats,        # (N, d)
        "labels": lbls,           # (N,)
        "logits": lgts,           # (N, K)
        "W": W,                   # (K, d)
        "split": args.split,
        "checkpoint": args.checkpoint,
    }, args.out)
    print(f"Saved {feats.shape[0]} features (d={feats.shape[1]}) to {args.out}")


if __name__ == "__main__":
    main()
