"""Graph construction utilities for the CivicTwin synthetic panel."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover
    Data = None


def build_graph(
    panel: np.ndarray,
    edge_index: Optional[np.ndarray] = None,
    no_edges: bool = False,
    feature_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Encode the panel as a graph object.

    If `no_edges` is true, the graph retains the node features but removes all
    adjacency edges, creating an isolated-node ablation.
    """
    if panel.ndim == 2:
        node_features = panel
    elif panel.ndim == 3:
        node_features = panel.reshape(panel.shape[0], -1)
    else:
        raise ValueError("panel must have shape (N, T, F) or (N, F)")

    if edge_index is None:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    if no_edges:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    if torch is not None and Data is not None:
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_tensor = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=x, edge_index=edge_tensor)
        data.num_nodes = x.size(0)
        data.node_features = x.clone()
        if feature_names is not None:
            data.feature_names = feature_names
        return data

    return {
        "x": node_features,
        "edge_index": edge_index,
        "node_features": panel,
        "feature_names": feature_names,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CivicTwin graph from a panel.")
    parser.add_argument("--no-edges", action="store_true", help="Strip the graph to isolated nodes.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"Graph builder initialized with no_edges={args.no_edges}")
