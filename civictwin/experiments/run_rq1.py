"""RQ1: does the graph-aware model beat non-graph baselines on spillover data?

The synthetic generator writes neighbour spillover explicitly into the panel, so
a graph-aware forecaster is the expected winner. This module runs the comparison
across multiple seeds and reports mean +/- std, because a single-seed win on a
16-node panel is noise, not evidence.

Pipeline per seed:

1. generate the synthetic city and select the spillover-bearing features
2. build rolling history windows, split temporally (no shuffling)
3. fit a z-score scaler **on the training windows only** and apply it to both
   splits, so land value (~1e2) cannot dominate accessibility (~1e0) in the
   squared-error objective
4. train the ST-GNN and the MLP; the persistence baseline is parameter-free
5. invert the scaling and score MAE / RMSE in the original units
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from civictwin.evaluate import (
    DEFAULT_DPI_WEIGHTS,
    DEFAULT_OVERBURDEN_PARAMS,
    compare_policies,
    regression_metrics,
)
from civictwin.model import MLPBaseline, PersistenceBaseline, SpatioTemporalGNN
from civictwin.policy import (
    baseline_policy,
    inclusionary_housing_policy,
    land_value_capture_policy,
    market_led_policy,
)
from civictwin.scaling import FeatureScaler
from civictwin.synth import generate_synthetic_city

#: Panel columns carrying the spillover dynamics: land value, rent, accessibility.
RQ1_FEATURE_COLUMNS = [0, 1, 5]
RQ1_FEATURE_NAMES = ["land_value", "rent", "accessibility"]
DEFAULT_SEEDS = (1, 2, 3, 4, 5)


def make_windows(panel: np.ndarray, window_size: int = 4) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Create rolling history windows and next-step targets for forecasting."""
    windows: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for t in range(window_size - 1, panel.shape[1] - 1):
        windows.append(panel[:, t - window_size + 1 : t + 1, :])
        targets.append(panel[:, t + 1, :])
    return windows, targets


def _is_trainable(model: torch.nn.Module) -> bool:
    return any(parameter.requires_grad for parameter in model.parameters())


def _predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    treatment: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatch a forward pass, passing the graph only to graph-aware models."""
    if isinstance(model, SpatioTemporalGNN):
        return model(x, edge_index.to(device), treatment=treatment)
    return model(x)


def train_model(
    model: torch.nn.Module,
    windows: List[np.ndarray],
    targets: List[np.ndarray],
    edge_index: torch.Tensor,
    device: torch.device,
    epochs: int = 25,
    learning_rate: float = 1e-3,
) -> None:
    """Train the model on the synthetic panel.

    Parameter-free models (e.g. persistence) are returned untouched.
    """
    if not _is_trainable(model):
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for window, target in zip(windows, targets):
            x = torch.tensor(window.astype(np.float32), device=device)
            y = torch.tensor(target.astype(np.float32), device=device)
            optimizer.zero_grad()
            pred = _predict(model, x, edge_index, device)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        if epoch_loss == 0.0:
            break


def evaluate_model(
    model: torch.nn.Module,
    windows: List[np.ndarray],
    targets: List[np.ndarray],
    edge_index: torch.Tensor,
    feature_names: List[str],
    device: torch.device,
    scaler: Optional[FeatureScaler] = None,
) -> Dict[str, float]:
    """Compute per-feature and overall MAE / RMSE.

    When ``scaler`` is given, predictions and targets are mapped back to the
    original units before scoring, so metrics stay comparable across runs with
    and without normalisation. Errors are pooled over every window, node and
    feature, which makes ``*_rmse`` a true root-mean-squared error rather than a
    root-mean-square of per-window absolute errors.
    """
    predictions: List[np.ndarray] = []
    observations: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for window, target in zip(windows, targets):
            x = torch.tensor(window.astype(np.float32), device=device)
            pred = _predict(model, x, edge_index, device)
            predictions.append(pred.detach().cpu().numpy())
            observations.append(np.asarray(target, dtype=np.float64))

    pred_stack = np.stack(predictions, axis=0).astype(np.float64)
    true_stack = np.stack(observations, axis=0)

    if scaler is not None:
        pred_stack = scaler.inverse_transform(pred_stack)
        true_stack = scaler.inverse_transform(true_stack)

    return regression_metrics(pred_stack, true_stack, feature_names)


def summarize_causal_split(
    model: SpatioTemporalGNN,
    window: np.ndarray,
    edge_index: torch.Tensor,
    treatment: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """Report the magnitude of the direct vs. interference channels.

    A near-zero interference magnitude on a graph with treated neighbours means
    the spillover channel is carrying no signal -- a diagnostic worth logging
    next to the forecast metrics.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(window.astype(np.float32), device=device)
        w = torch.tensor(treatment.astype(np.float32), device=device)
        parts = model.decompose(x, edge_index.to(device), treatment=w)

    return {
        "direct_effect_l1": float(parts.direct.abs().mean().item()),
        "interference_effect_l1": float(parts.interference.abs().mean().item()),
        "treated_fraction": float(parts.treatment.mean().item()),
        "mean_exposure": float(parts.exposure.mean().item()),
    }


def policy_boundary(n_neighborhoods: int, config: Optional[Dict[str, Any]] = None) -> List[int]:
    """Treated node set: the accessibility-shock nodes present in this city."""
    synth_cfg = (config or {}).get("synth", {})
    shock_nodes = synth_cfg.get("accessibility_shock_nodes", [0, 1, 4, 7])
    boundary = sorted({int(node) for node in shock_nodes if 0 <= int(node) < n_neighborhoods})
    return boundary or [0]


def treatment_vector(n_neighborhoods: int, boundary: Sequence[int]) -> np.ndarray:
    """Binary node-level treatment indicator ``w`` of shape ``(N,)``."""
    w = np.zeros(n_neighborhoods, dtype=np.float64)
    w[list(boundary)] = 1.0
    return w


def build_policy_scorecard(
    city: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    overburden_params: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Score the four RQ1 scenarios against the untreated baseline panel.

    Intensities come from ``config["policy"]``; the treated boundary is the
    shock-node set. Returns one row per scenario with affordability, DPI and
    H+T overburden deltas plus an ascending pressure ranking.
    """
    panel = city["panel"]
    config = city.get("config", {}) or {}
    n_neighborhoods = int(city.get("n_neighborhoods", panel.shape[0]))
    boundary = policy_boundary(n_neighborhoods, config)

    policy_cfg = config.get("policy", {}) or {}

    def intensity(name: str, fallback: float) -> float:
        entry = policy_cfg.get(name, {}) or {}
        return float(entry.get("intensity", fallback))

    policies = [
        baseline_policy(),
        market_led_policy(boundary, timing=2, intensity=intensity("market_led", 0.8)),
        inclusionary_housing_policy(
            boundary, timing=1, intensity=intensity("inclusionary_housing", 0.6)
        ),
        land_value_capture_policy(
            boundary, timing=2, intensity=intensity("land_value_capture", 0.7)
        ),
    ]

    if weights is None:
        weights = dict((config.get("scoring", {}) or {}).get("weights", DEFAULT_DPI_WEIGHTS))
    if overburden_params is None:
        overburden_params = dict(
            (config.get("scoring", {}) or {}).get("overburden", DEFAULT_OVERBURDEN_PARAMS)
        )

    return compare_policies(panel, policies, weights=weights, overburden_params=overburden_params)


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolve ``auto`` / ``cpu`` / ``cuda`` into a concrete device."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def run_forecast_experiment(
    seed: int = 42,
    n_neighborhoods: int = 16,
    n_steps: int = 18,
    epochs: int = 30,
    config: Optional[Dict[str, Any]] = None,
    window_size: int = 4,
    hidden_dim: int = 12,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
    scale_features: bool = True,
) -> Dict[str, float]:
    """Run one seeded RQ1 comparison and return a flat metrics row.

    Models compared: the ST-GNN, an MLP that ignores the graph, and a naive
    persistence forecaster. ``baseline_*`` keys alias the MLP metrics, which is
    the non-graph reference RQ1 is stated against.
    """
    torch_device = resolve_device(device)
    torch.manual_seed(seed)

    city = generate_synthetic_city(
        seed=seed, n_neighborhoods=n_neighborhoods, n_steps=n_steps, config=config
    )
    panel = city["panel"][:, :, RQ1_FEATURE_COLUMNS]
    edge_index = torch.tensor(city["edge_index"], dtype=torch.long)
    n_features = len(RQ1_FEATURE_NAMES)

    windows, targets = make_windows(panel, window_size=window_size)
    if len(windows) < 2:
        raise ValueError(
            f"n_steps={n_steps} with window_size={window_size} yields {len(windows)} windows; "
            "at least 2 are required for a train/test split"
        )

    split = max(1, int(len(windows) * 0.7))
    train_windows, train_targets = windows[:split], targets[:split]
    test_windows, test_targets = windows[split:], targets[split:]
    if not test_windows:  # pragma: no cover - guarded by the length check above
        train_windows, train_targets = windows[:-1], targets[:-1]
        test_windows, test_targets = windows[-1:], targets[-1:]

    # Fit on the training split only: the scaler must not see the future.
    scaler: Optional[FeatureScaler] = None
    if scale_features:
        scaler = FeatureScaler.fit_windows(train_windows, feature_names=RQ1_FEATURE_NAMES)
        train_windows = [scaler.transform(w) for w in train_windows]
        train_targets = [scaler.transform(t) for t in train_targets]
        test_windows = [scaler.transform(w) for w in test_windows]
        test_targets = [scaler.transform(t) for t in test_targets]

    models: Dict[str, torch.nn.Module] = {
        "graph": SpatioTemporalGNN(
            input_dim=n_features, hidden_dim=hidden_dim, output_dim=n_features
        ),
        "mlp": MLPBaseline(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            output_dim=n_features,
            sequence_length=window_size,
        ),
        "persistence": PersistenceBaseline(output_dim=n_features),
    }

    row: Dict[str, float] = {"seed": int(seed)}
    for name, model in models.items():
        model.to(torch_device)
        train_model(
            model,
            train_windows,
            train_targets,
            edge_index,
            device=torch_device,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        metrics = evaluate_model(
            model,
            test_windows,
            test_targets,
            edge_index,
            RQ1_FEATURE_NAMES,
            torch_device,
            scaler=scaler,
        )
        for key, value in metrics.items():
            row[f"{name}_{key}"] = value

    # RQ1 is stated against the non-graph learned baseline.
    row["baseline_overall_mae"] = row["mlp_overall_mae"]
    row["baseline_overall_rmse"] = row["mlp_overall_rmse"]
    row["graph_beats_baseline"] = float(row["graph_overall_mae"] < row["mlp_overall_mae"])

    treatment = treatment_vector(
        panel.shape[0], policy_boundary(panel.shape[0], city.get("config", {}))
    )
    row.update(
        summarize_causal_split(
            models["graph"], test_windows[-1], edge_index, treatment, torch_device
        )
    )
    return row


def run_multi_seed_benchmark(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run :func:`run_forecast_experiment` across seeds.

    Returns ``(per_seed, summary)`` where ``summary`` holds the mean, std, min
    and max of every numeric metric. Standard deviation uses the sample
    convention (ddof=1) and is NaN for a single seed, which is the honest
    reading rather than zero.
    """
    if len(seeds) == 0:
        raise ValueError("at least one seed is required")

    rows = [run_forecast_experiment(seed=int(seed), **kwargs) for seed in seeds]
    per_seed = pd.DataFrame(rows)

    metric_columns = [column for column in per_seed.columns if column != "seed"]
    summary = pd.DataFrame(
        {
            "metric": metric_columns,
            "mean": [per_seed[column].mean() for column in metric_columns],
            "std": [per_seed[column].std(ddof=1) for column in metric_columns],
            "min": [per_seed[column].min() for column in metric_columns],
            "max": [per_seed[column].max() for column in metric_columns],
            "n_seeds": len(per_seed),
        }
    )
    return per_seed, summary


def _write_csv(frame: pd.DataFrame, path: Optional[str]) -> None:
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"wrote {destination}")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RQ1 multi-seed benchmark: ST-GNN vs. MLP vs. persistence."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-neighborhoods", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--no-scale", action="store_true", help="Disable z-score feature normalisation."
    )
    parser.add_argument("--output", default="artifacts/rq1.csv", help="Per-seed metrics CSV.")
    parser.add_argument(
        "--summary-output", default="artifacts/rq1_summary.csv", help="Mean/std summary CSV."
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Report the RQ1 verdict instead of exiting non-zero when the graph model loses.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    device = resolve_device(args.device)
    print(f"RQ1 benchmark on {device} over seeds {args.seeds}")

    per_seed, summary = run_multi_seed_benchmark(
        seeds=args.seeds,
        n_neighborhoods=args.n_neighborhoods,
        n_steps=args.n_steps,
        epochs=args.epochs,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        device=args.device,
        scale_features=not args.no_scale,
    )

    print("\nPer-seed overall MAE:")
    for _, row in per_seed.iterrows():
        print(
            f"  seed={int(row['seed'])}: graph={row['graph_overall_mae']:.4f}, "
            f"mlp={row['mlp_overall_mae']:.4f}, persistence={row['persistence_overall_mae']:.4f}"
        )

    print(f"\nAggregate over {len(per_seed)} seeds (mean +/- std):")
    indexed = summary.set_index("metric")
    for model in ("graph", "mlp", "persistence"):
        for metric in ("overall_mae", "overall_rmse"):
            key = f"{model}_{metric}"
            print(f"  {key}: {indexed.loc[key, 'mean']:.4f} +/- {indexed.loc[key, 'std']:.4f}")

    print("\nCausal channel magnitudes (ST-GNN, treated boundary):")
    for key in ("direct_effect_l1", "interference_effect_l1", "mean_exposure"):
        print(f"  {key}: {indexed.loc[key, 'mean']:.4f} +/- {indexed.loc[key, 'std']:.4f}")

    scorecard = build_policy_scorecard(
        generate_synthetic_city(
            seed=int(args.seeds[0]),
            n_neighborhoods=args.n_neighborhoods,
            n_steps=args.n_steps,
        )
    )
    print("\nPolicy scorecard (seed {}):".format(args.seeds[0]))
    print(scorecard.to_string(index=False))

    _write_csv(per_seed, args.output)
    _write_csv(summary, args.summary_output)

    graph_mae = float(per_seed["graph_overall_mae"].mean())
    mlp_mae = float(per_seed["mlp_overall_mae"].mean())
    wins = int(per_seed["graph_beats_baseline"].sum())
    verdict = "PASS" if graph_mae < mlp_mae else "FAIL"
    print(
        f"\nRQ1 verdict: {verdict} -- graph mean MAE {graph_mae:.4f} vs. baseline {mlp_mae:.4f} "
        f"({wins}/{len(per_seed)} seeds won)"
    )

    if verdict == "FAIL" and not args.no_strict:
        raise RuntimeError(
            "RQ1 failed: the graph model did not outperform the non-graph baseline "
            "on the synthetic spillover task (mean over seeds)."
        )


if __name__ == "__main__":
    main()
