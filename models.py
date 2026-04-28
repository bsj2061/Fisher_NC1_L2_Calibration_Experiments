"""
ResNet-18 model adapted for CIFAR-100, with explicit access to penultimate features.

Key design points:
- The classifier head is a single Linear layer W: R^d -> R^K (no bias for clean ETF analysis).
- forward() returns both logits and penultimate features.
- This matches the PDF's UFM setup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18Cifar(nn.Module):
    """
    ResNet-18 for CIFAR (32x32 inputs).

    The classifier W has shape (K, d) with NO bias term. This makes the simplex ETF
    analysis cleaner -- ETF self-duality (NC3) implies W is proportional to the
    matrix of class means, which assumes no bias.
    """

    def __init__(self, num_classes=100, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        # Single linear classifier, no bias. Shape (K, d).
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        """Return penultimate features (before classifier)."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)  # (B, d)
        return out

    def forward(self, x, return_features=False):
        h = self.features(x)
        logits = self.classifier(h)
        if return_features:
            return logits, h
        return logits

    @property
    def W(self):
        """Classifier weight matrix W of shape (K, d)."""
        return self.classifier.weight.data


def get_model(num_classes=100, device="cuda"):
    return ResNet18Cifar(num_classes=num_classes).to(device)
