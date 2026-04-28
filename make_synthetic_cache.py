"""
Make a synthetic feature cache file in the same format as extract_features.py
produces, for testing exp1/2/3 scripts without training a real model.
"""
import env_setup  # noqa: F401  -- OpenMP workaround, must be first

import os
import torch
from test_e2e_synthetic import build_synthetic_dataset

K, d = 10, 64
feats, labs, logits, W, mu = build_synthetic_dataset(
    K=K, d=d, n_per=200, rho=2.0, noise_row=0.6, noise_null=1.5, seed=42
)

os.makedirs("cache", exist_ok=True)
torch.save({
    "features": feats,
    "labels": labs,
    "logits": logits,
    "W": W,
    "split": "synthetic",
    "checkpoint": "synthetic",
}, "cache/synthetic.pt")
print(f"Saved synthetic cache: {feats.shape[0]} samples, K={K}, d={d}")
