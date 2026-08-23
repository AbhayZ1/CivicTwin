"""Spatio-temporal forecasting models for CivicTwin."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

try:
    from torch_geometric.nn import GCNConv
except ImportError:  # pragma: no cover
    GCNConv = None


class SpatioTemporalGNN(nn.Module):
    """Simple graph-temporal model: GRU over time plus a GCN-style spatial layer."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 16,
        output_dim: int = 6,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        if GCNConv is not None:
            self.spatial = GCNConv(hidden_dim, hidden_dim)
        else:
            self.spatial = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x_seq shape: [N, T, F]. Returns predictions for t+1 at each node."""
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)

        gru_out, _ = self.gru(x_seq)
        hidden = gru_out[:, -1, :]

        if edge_index is not None and edge_index.numel() > 0:
            if GCNConv is not None:
                hidden = self.spatial(hidden, edge_index)
            else:
                hidden = self.spatial(hidden)
        else:
            hidden = self.spatial(hidden)

        hidden = self.dropout(hidden)
        return self.head(hidden)


class MLPBaseline(nn.Module):
    """Simple baseline that ignores graph structure and sees only node history."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 16, output_dim: int = 6, sequence_length: int = 5) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.net = nn.Sequential(
            nn.Linear(input_dim * sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)
        flattened = x_seq[:, -self.sequence_length :, :].reshape(x_seq.size(0), -1)
        return self.net(flattened)


__all__ = ["SpatioTemporalGNN", "MLPBaseline"]
