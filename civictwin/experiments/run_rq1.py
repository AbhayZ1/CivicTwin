"""RQ1 experiment: test whether the graph model beats the no-graph baseline.

The synthetic generator is built so that neighbor spillover is explicit in the data,
which makes a graph-aware model the expected winner. The script trains both models
on earlier steps and evaluates them on a held-out window.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from civictwin.model import MLPBaseline, SpatioTemporalGNN
from civictwin.synth import generate_synthetic_city


def make_windows(panel: np.ndarray, window_size: int = 4) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Create rolling history windows and next-step targets for forecasting."""
    windows: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for t in range(window_size - 1, panel.shape[1] - 1):
        windows.append(panel[:, t - window_size + 1 : t + 1, :])
        targets.append(panel[:, t + 1, :])
    return windows, targets


def evaluate_model(
    model: torch.nn.Module,
    windows: List[np.ndarray],
    targets: List[np.ndarray],
    edge_index: torch.Tensor,
    feature_names: List[str],
    device: torch.device,
) -> Dict[str, float]:
    """Compute MAE and RMSE for each feature and the overall mean across features."""
    all_errors: Dict[str, List[float]] = {name: [] for name in feature_names}
    model.eval()
    with torch.no_grad():
        for window, target in zip(windows, targets):
            x = torch.tensor(window.astype(np.float32), device=device)
            y = torch.tensor(target.astype(np.float32), device=device)
            if isinstance(model, SpatioTemporalGNN):
                pred = model(x, edge_index.to(device))
            else:
                pred = model(x)
            abs_error = (pred - y).abs()
            for idx, name in enumerate(feature_names):
                all_errors[name].append(float(abs_error[:, idx].mean().item()))

    metrics = {}
    for name, values in all_errors.items():
        arr = np.asarray(values, dtype=np.float64)
        metrics[f"{name}_mae"] = float(arr.mean())
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean(arr ** 2)))

    overall_mae = float(np.mean([v for k, v in metrics.items() if k.endswith("_mae")]))
    overall_rmse = float(
        np.sqrt(np.mean([v ** 2 for k, v in metrics.items() if k.endswith("_rmse")]))
    )
    metrics["overall_mae"] = overall_mae
    metrics["overall_rmse"] = overall_rmse
    return metrics


def train_model(
    model: torch.nn.Module,
    windows: List[np.ndarray],
    targets: List[np.ndarray],
    edge_index: torch.Tensor,
    device: torch.device,
    epochs: int = 25,
    learning_rate: float = 1e-3,
) -> None:
    """Train the model for a few epochs on the synthetic panel."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for window, target in zip(windows, targets):
            x = torch.tensor(window.astype(np.float32), device=device)
            y = torch.tensor(target.astype(np.float32), device=device)
            optimizer.zero_grad()
            if isinstance(model, SpatioTemporalGNN):
                pred = model(x, edge_index.to(device))
            else:
                pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        if epoch_loss == 0.0:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare the graph model against a no-graph baseline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighborhoods", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    city = generate_synthetic_city(seed=args.seed, n_neighborhoods=args.n_neighborhoods, n_steps=args.n_steps)
    panel = city["panel"]
    feature_names = city["feature_names"]
    edge_index = torch.tensor(city["edge_index"], dtype=torch.long)

    windows, targets = make_windows(panel, window_size=4)
    split = max(1, int(len(windows) * 0.7))
    train_windows, train_targets = windows[:split], targets[:split]
    test_windows, test_targets = windows[split:], targets[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_model = SpatioTemporalGNN(input_dim=len(feature_names), hidden_dim=8, output_dim=len(feature_names))
    mlp_model = MLPBaseline(input_dim=len(feature_names), hidden_dim=8, output_dim=len(feature_names), sequence_length=4)
    graph_model.to(device)
    mlp_model.to(device)

    train_model(graph_model, train_windows, train_targets, edge_index, device=device, epochs=args.epochs)
    train_model(mlp_model, train_windows, train_targets, edge_index, device=device, epochs=args.epochs)

    graph_metrics = evaluate_model(graph_model, test_windows, test_targets, edge_index, feature_names, device)
    mlp_metrics = evaluate_model(mlp_model, test_windows, test_targets, edge_index, feature_names, device)

    print("RQ1: Graph model vs baseline")
    print("Feature-wise MAE/RMSE:")
    for name in feature_names:
        print(f"  {name}: graph_mae={graph_metrics[f'{name}_mae']:.4f}, graph_rmse={graph_metrics[f'{name}_rmse']:.4f}, mlp_mae={mlp_metrics[f'{name}_mae']:.4f}, mlp_rmse={mlp_metrics[f'{name}_rmse']:.4f}")

    print(f"Overall graph MAE={graph_metrics['overall_mae']:.4f}, RMSE={graph_metrics['overall_rmse']:.4f}")
    print(f"Overall baseline MAE={mlp_metrics['overall_mae']:.4f}, RMSE={mlp_metrics['overall_rmse']:.4f}")

    if graph_metrics["overall_mae"] >= mlp_metrics["overall_mae"]:
        raise RuntimeError("RQ1 failed: graph model did not outperform the non-graph baseline on the synthetic spillover task.")


if __name__ == "__main__":
    main()
