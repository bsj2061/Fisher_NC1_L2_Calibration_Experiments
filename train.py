"""
Minimal training script for ResNet-18 on CIFAR-100.

Run:
    python train.py --epochs 30 --batch-size 128 --lr 1e-3 \
                    --save-dir checkpoints/baseline

Notes:
- We use AdamW + cosine schedule to roughly match the PDF's setup.
- Default 30 epochs is enough to reach ~60% accuracy and observe NC trends.
- For best calibration analysis, 100+ epochs makes NC stronger.
"""
from __future__ import annotations
import env_setup  # noqa: F401  -- OpenMP workaround, must be first

import argparse
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from models import get_model


def get_loaders(batch_size: int, data_root: str = "./data", num_workers: int = 4):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    train_set = torchvision.datasets.CIFAR100(data_root, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR100(data_root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return {"acc": correct / total, "loss": loss_sum / total}


def train_one_epoch(model, loader, optimizer, device, log_interval=100):
    model.train()
    correct, total, loss_sum = 0, 0, 0.0
    for i, (x, y) in enumerate(loader):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="checkpoints/baseline")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, test_loader = get_loaders(args.batch_size, args.data_root)
    model = get_model(num_classes=args.num_classes, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_acc = 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, args.device)
        test_stats = evaluate(model, test_loader, args.device)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train loss {train_stats['loss']:.3f} acc {train_stats['acc']:.3f} | "
              f"test loss {test_stats['loss']:.3f} acc {test_stats['acc']:.3f} | "
              f"{elapsed:.0f}s")
        history.append({
            "epoch": epoch + 1,
            "train": train_stats,
            "test": test_stats,
            "lr": scheduler.get_last_lr()[0],
        })
        if test_stats["acc"] > best_acc:
            best_acc = test_stats["acc"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))
        torch.save(model.state_dict(), os.path.join(args.save_dir, "last.pt"))

    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump({"args": vars(args), "history": history, "best_test_acc": best_acc}, f, indent=2)
    print(f"Done. Best test acc: {best_acc:.4f}. Saved to {args.save_dir}")


if __name__ == "__main__":
    main()
