"""Spatio-temporal forecasting models with an explicit causal decomposition.

Estimand
--------
Let ``w_i in {0, 1}`` mark whether neighbourhood ``i`` receives a policy
intervention and let ``A`` be the neighbourhood adjacency. Under *partial
interference* (Hudgens & Halloran), a unit outcome depends on its own treatment
and on an exposure summary of its neighbours treatments:

    e_i  =  ( sum_j A_ij w_j ) / max(deg_i, 1)          (neighbour exposure)

The model factorises the one-step-ahead prediction into three additive parts:

    Y_hat_i  =  mu_i            (counterfactual / untreated outcome)
             +  tau_i   * w_i   (DIRECT policy effect,     node-level)
             +  gamma_i * e_i   (SPATIAL INTERFERENCE,     neighbour spillover)

Identification of the split is architectural rather than statistical:

* ``mu`` reads both streams -- absent any policy the graph still carries the
  synthetic spillover dynamics, so the counterfactual is allowed to be
  graph-aware.
* ``tau`` reads the **node-local** stream only. It never sees a message-passing
  operator, so by construction it cannot absorb neighbour information.
* ``gamma`` reads the **message-passed** stream, so the spillover channel is the
  only route by which neighbour state can enter the treated response.

With ``w = 0`` the model reduces exactly to ``mu``, i.e. the plain forecasting
behaviour used by RQ1. With no edges, ``e = 0`` and the interference term
vanishes -- the no-edge ablation is therefore a strict special case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn

try:
    from torch_geometric.nn import GCNConv
except ImportError:  # pragma: no cover
    GCNConv = None


@dataclass
class CausalDecomposition:
    """Isolated components of a single forward pass.

    All tensors are shaped ``[N, output_dim]`` except ``treatment`` and
    ``exposure`` which are ``[N, 1]``. By construction
    ``prediction == baseline + direct + interference``.
    """

    prediction: torch.Tensor
    baseline: torch.Tensor
    direct: torch.Tensor
    interference: torch.Tensor
    treatment: torch.Tensor
    exposure: torch.Tensor

    @property
    def total_effect(self) -> torch.Tensor:
        """Total policy effect relative to the untreated counterfactual."""
        return self.direct + self.interference

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "prediction": self.prediction,
            "baseline": self.baseline,
            "direct": self.direct,
            "interference": self.interference,
            "total_effect": self.total_effect,
            "treatment": self.treatment,
            "exposure": self.exposure,
        }


def neighbor_exposure(
    treatment: torch.Tensor,
    edge_index: Optional[torch.Tensor],
    num_nodes: int,
) -> torch.Tensor:
    """Degree-normalised neighbour treatment exposure ``e_i``.

    ``treatment`` is ``[N]`` or ``[N, 1]``; the return is ``[N, 1]``. Isolated
    nodes and empty graphs yield zero exposure, never NaN.
    """
    flat = treatment.reshape(-1)
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros(num_nodes, 1, dtype=flat.dtype, device=flat.device)

    src, dst = edge_index[0], edge_index[1]
    summed = torch.zeros(num_nodes, dtype=flat.dtype, device=flat.device)
    summed.index_add_(0, dst, flat.index_select(0, src))
    degree = torch.zeros(num_nodes, dtype=flat.dtype, device=flat.device)
    degree.index_add_(0, dst, torch.ones_like(dst, dtype=flat.dtype))
    return (summed / degree.clamp(min=1.0)).unsqueeze(-1)


class SpatioTemporalGNN(nn.Module):
    """Two-stream ST-GNN that decomposes direct and interference effects.

    Stream A (local):   encoder -> GRU                     -> h_local
    Stream B (spatial): encoder -> GCN (per step) -> GRU   -> h_spatial

    ``forward`` returns a plain tensor by default so the module stays a drop-in
    forecaster; pass ``return_components=True`` (or call :meth:`decompose`) to
    recover the isolated causal signals.
    """

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

        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.dropout = nn.Dropout(dropout)

        if GCNConv is not None:
            self.spatial = GCNConv(hidden_dim, hidden_dim)
        else:  # pragma: no cover - exercised only without torch-geometric
            self.spatial = nn.Linear(hidden_dim, hidden_dim)
        # Used when the graph has no edges, keeping both streams well defined.
        self.spatial_identity = nn.Linear(hidden_dim, hidden_dim)

        self.gru_local = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.gru_spatial = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # mu: counterfactual outcome, may use both streams.
        self.baseline_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )
        # tau: direct effect, node-local stream ONLY (no message passing).
        self.direct_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )
        # gamma: interference effect, message-passed stream ONLY.
        self.interference_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _mix(self, step_features: torch.Tensor, edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return self.spatial_identity(step_features)
        if GCNConv is not None:
            return self.spatial(step_features, edge_index)
        return self.spatial(step_features)  # pragma: no cover

    def _encode(
        self, x_seq: torch.Tensor, edge_index: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run both temporal streams, returning ``(h_local, h_spatial)``."""
        local_steps = []
        spatial_steps = []
        for t in range(x_seq.size(1)):
            encoded = self.encoder(x_seq[:, t, :])
            local_steps.append(self.dropout(encoded))
            spatial_steps.append(self.dropout(self._mix(encoded, edge_index)))

        local_out, _ = self.gru_local(torch.stack(local_steps, dim=1))
        spatial_out, _ = self.gru_spatial(torch.stack(spatial_steps, dim=1))
        return local_out[:, -1, :], spatial_out[:, -1, :]

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x_seq: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        treatment: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, CausalDecomposition]:
        """``x_seq``: ``[N, T, F]``. Returns the ``t+1`` prediction per node.

        ``treatment`` is an optional ``[N]`` / ``[N, 1]`` policy indicator; when
        omitted the prediction is the untreated counterfactual ``mu``.
        """
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)

        num_nodes = x_seq.size(0)
        h_local, h_spatial = self._encode(x_seq, edge_index)

        if treatment is None:
            w = torch.zeros(num_nodes, 1, dtype=x_seq.dtype, device=x_seq.device)
        else:
            w = treatment.reshape(num_nodes, 1).to(dtype=x_seq.dtype, device=x_seq.device)

        exposure = neighbor_exposure(w, edge_index, num_nodes)

        baseline = self.baseline_head(torch.cat([h_local, h_spatial], dim=-1))
        direct = self.direct_head(h_local) * w
        interference = self.interference_head(h_spatial) * exposure
        prediction = baseline + direct + interference

        if not return_components:
            return prediction

        return CausalDecomposition(
            prediction=prediction,
            baseline=baseline,
            direct=direct,
            interference=interference,
            treatment=w,
            exposure=exposure,
        )

    def decompose(
        self,
        x_seq: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        treatment: Optional[torch.Tensor] = None,
    ) -> CausalDecomposition:
        """Isolate ``mu``, the direct effect and the interference effect."""
        result = self.forward(x_seq, edge_index, treatment=treatment, return_components=True)
        assert isinstance(result, CausalDecomposition)  # narrows the Union for type checkers
        return result

    def direct_effect(
        self,
        x_seq: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        treatment: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Direct policy effect ``tau_i * w_i`` in isolation."""
        return self.decompose(x_seq, edge_index, treatment).direct

    def interference_effect(
        self,
        x_seq: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        treatment: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Spatial interference (spillover) effect ``gamma_i * e_i`` in isolation."""
        return self.decompose(x_seq, edge_index, treatment).interference


class MLPBaseline(nn.Module):
    """Non-graph baseline: sees a flattened node history and nothing else."""

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
        history = x_seq[:, -self.sequence_length :, :]
        if history.size(1) < self.sequence_length:
            pad = history[:, :1, :].expand(-1, self.sequence_length - history.size(1), -1)
            history = torch.cat([pad, history], dim=1)
        return self.net(history.reshape(x_seq.size(0), -1))


class PersistenceBaseline(nn.Module):
    """Naive forecaster: ``y_hat_{t+1} = y_t``.

    Parameter-free, so it is never trained. It is the reference every learned
    model must beat before any claim about spillover structure is meaningful.
    """

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
    "MLPBaseline",
    "PersistenceBaseline",
    "CausalDecomposition",
    "neighbor_exposure",
]
