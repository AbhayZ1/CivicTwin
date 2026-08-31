from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn


@dataclass
class CausalDecomposition:
    prediction: torch.Tensor
    direct: torch.Tensor
    spillover: torch.Tensor
    treatment: torch.Tensor
    exposure: torch.Tensor

    @property
    def baseline(self) -> torch.Tensor:
        return self.direct

    @property
    def interference(self) -> torch.Tensor:
        return self.spillover

    @property
    def total_effect(self) -> torch.Tensor:
        return self.direct + self.spillover

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "prediction": self.prediction,
            "direct": self.direct,
            "spillover": self.spillover,
            "total_effect": self.total_effect,
            "treatment": self.treatment,
            "exposure": self.exposure,
        }


def normalized_adjacency(
    edge_index: Optional[torch.Tensor],
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
    edge_weight: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if edge_index is None or edge_index.numel() == 0:
        return None
    src = edge_index[0].to(device)
    dst = edge_index[1].to(device)
    if edge_weight is None:
        weight = torch.ones(src.numel(), device=device, dtype=dtype)
    else:
        weight = edge_weight.reshape(-1).to(device=device, dtype=dtype)
    degree = torch.zeros(num_nodes, device=device, dtype=dtype)
    degree.index_add_(0, dst, weight)
    return src, dst, weight / degree.clamp(min=1e-12).index_select(0, dst)


def aggregate_neighbors(
    values: torch.Tensor,
    edge_index: Optional[torch.Tensor],
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = torch.zeros(num_nodes, values.size(-1), device=values.device, dtype=values.dtype)
    parts = normalized_adjacency(
        edge_index, num_nodes, values.device, values.dtype, edge_weight
    )
    if parts is None:
        return out
    src, dst, weight = parts
    out.index_add_(0, dst, values.index_select(0, src) * weight.unsqueeze(-1))
    return out


def neighbor_exposure(
    treatment: torch.Tensor,
    edge_index: Optional[torch.Tensor],
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return aggregate_neighbors(
        treatment.reshape(num_nodes, 1), edge_index, num_nodes, edge_weight
    )


class SpatioTemporalGNN(nn.Module):
    uses_graph = True

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 16,
        output_dim: int = 6,
        num_layers: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.residual = bool(residual) and output_dim <= input_dim

        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.phi = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )
        self.psi = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def encode(self, x_seq: torch.Tensor) -> torch.Tensor:
        steps = [self.dropout(self.encoder(x_seq[:, t, :])) for t in range(x_seq.size(1))]
        out, _ = self.gru(torch.stack(steps, dim=1))
        return out[:, -1, :]

    def _terms(
        self,
        hidden: torch.Tensor,
        treatment: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        num_nodes: int,
        edge_weight: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = torch.cat([hidden, treatment], dim=-1)
        direct = self.phi(state)
        spillover = aggregate_neighbors(self.psi(state), edge_index, num_nodes, edge_weight)
        return direct, spillover

    def forward(
        self,
        x_seq: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        treatment: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, CausalDecomposition]:
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)

        num_nodes = x_seq.size(0)
        hidden = self.encode(x_seq)

        if treatment is None:
            w = torch.zeros(num_nodes, 1, dtype=x_seq.dtype, device=x_seq.device)
        else:
            w = treatment.reshape(num_nodes, 1).to(dtype=x_seq.dtype, device=x_seq.device)

        direct, spillover = self._terms(hidden, w, edge_index, num_nodes, edge_weight)
        if self.residual:
            direct = direct + x_seq[:, -1, : self.output_dim]
        prediction = direct + spillover

        if not return_components:
            return prediction

        return CausalDecomposition(
            prediction=prediction,
            direct=direct,
            spillover=spillover,
            treatment=w,
            exposure=neighbor_exposure(w, edge_index, num_nodes, edge_weight),
        )

    def decompose(
        self,
        x_seq: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        treatment: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> CausalDecomposition:
        result = self.forward(
            x_seq, edge_index, treatment=treatment, edge_weight=edge_weight,
            return_components=True,
        )
        assert isinstance(result, CausalDecomposition)
        return result

    def causal_effects(
        self,
        x_seq: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        treatment: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)
        num_nodes = x_seq.size(0)
        hidden = self.encode(x_seq)

        if treatment is None:
            w = torch.zeros(num_nodes, 1, dtype=x_seq.dtype, device=x_seq.device)
        else:
            w = treatment.reshape(num_nodes, 1).to(dtype=x_seq.dtype, device=x_seq.device)
        zero = torch.zeros_like(w)

        treated_direct, treated_spill = self._terms(
            hidden, w, edge_index, num_nodes, edge_weight
        )
        control_direct, control_spill = self._terms(
            hidden, zero, edge_index, num_nodes, edge_weight
        )
        return {
            "ite": treated_direct - control_direct,
            "ste": treated_spill - control_spill,
            "total": (treated_direct + treated_spill) - (control_direct + control_spill),
        }

    def individual_treatment_effect(self, *args, **kwargs) -> torch.Tensor:
        return self.causal_effects(*args, **kwargs)["ite"]

    def spatial_interference_effect(self, *args, **kwargs) -> torch.Tensor:
        return self.causal_effects(*args, **kwargs)["ste"]

    def direct_effect(self, *args, **kwargs) -> torch.Tensor:
        return self.decompose(*args, **kwargs).direct

    def interference_effect(self, *args, **kwargs) -> torch.Tensor:
        return self.decompose(*args, **kwargs).spillover


CausalSTGNN = SpatioTemporalGNN


class MLPBaseline(nn.Module):
    uses_graph = False

    def __init__(
        self, input_dim: int = 6, hidden_dim: int = 16, output_dim: int = 6, sequence_length: int = 5
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.net = nn.Sequential(
            nn.Linear(input_dim * sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _history(self, x_seq: torch.Tensor) -> torch.Tensor:
        history = x_seq[:, -self.sequence_length :, :]
        if history.size(1) < self.sequence_length:
            pad = history[:, :1, :].expand(-1, self.sequence_length - history.size(1), -1)
            history = torch.cat([pad, history], dim=1)
        return history

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)
        return self.net(self._history(x_seq).reshape(x_seq.size(0), -1))


class SpatialLagModel(nn.Module):
    uses_graph = True

    def __init__(
        self, input_dim: int = 6, output_dim: int = 6, sequence_length: int = 5
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.beta = nn.Linear(input_dim * sequence_length, output_dim)
        self.rho = nn.Parameter(torch.zeros(output_dim))

    def _history(self, x_seq: torch.Tensor) -> torch.Tensor:
        history = x_seq[:, -self.sequence_length :, :]
        if history.size(1) < self.sequence_length:
            pad = history[:, :1, :].expand(-1, self.sequence_length - history.size(1), -1)
            history = torch.cat([pad, history], dim=1)
        return history

    def forward(
        self,
        x_seq: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)
        num_nodes = x_seq.size(0)
        history = self._history(x_seq)
        trend = self.beta(history.reshape(num_nodes, -1))
        lag = aggregate_neighbors(
            x_seq[:, -1, : self.output_dim], edge_index, num_nodes, edge_weight
        )
        return trend + self.rho * lag + x_seq[:, -1, : self.output_dim]


class PersistenceBaseline(nn.Module):
    uses_graph = False

    def __init__(self, output_dim: Optional[int] = None) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)
        last = x_seq[:, -1, :]
        if self.output_dim is not None:
            last = last[:, : self.output_dim]
        return last


__all__ = [
    "SpatioTemporalGNN",
    "CausalSTGNN",
    "MLPBaseline",
    "SpatialLagModel",
    "PersistenceBaseline",
    "CausalDecomposition",
    "neighbor_exposure",
    "aggregate_neighbors",
    "normalized_adjacency",
]
