"""Classifier head for Stage A."""

from __future__ import annotations

import torch
from torch import nn


class StageAClassifierHead(nn.Module):
    """Two-layer MLP classifier head."""

    def __init__(self, hidden_dim: int, num_targets: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project pooled features to target logits."""

        return self.layers(features)
