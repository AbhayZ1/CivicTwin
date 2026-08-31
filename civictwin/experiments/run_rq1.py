from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from civictwin.evaluate import (
    DEFAULT_DPI_WEIGHTS,
    DEFAULT_HT_PARAMS,
    compare_policies,
    regression_metrics,
)
from civictwin.model import (
    MLPBaseline,
    PersistenceBaseline,
    SpatialLagModel,
    SpatioTemporalGNN,
)
from civictwin.policy import (
    Policy,
    apply_policy,
    baseline_policy,
    inclusionary_housing_policy,
    land_value_capture_policy,
    market_led_policy,
)
from civictwin.scaling import FeatureScaler
from civictwin.synth import generate_synthetic_city

RQ1_FEATURE_COLUMNS = [0, 1, 5]
RQ1_FEATURE_NAMES = ["land_value", "rent", "accessibility"]
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
SCENARIOS = ("baseline", "market_led", "inclusionary_housing", "land_value_capture")


def make_windows(panel: np.ndarray, window_size: int = 4) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
    if not getattr(model, "uses_graph", False):
        return model(x)
    if isinstance(model, SpatioTemporalGNN):
        return model(x, edge_index.to(device), treatment=treatment)
    return model(x, edge_index.to(device))


def train_model(
    model: torch.nn.Module,
    windows: List[np.ndarray],
    targets: List[np.ndarray],
    edge_index: torch.Tensor,
    device: torch.device,
    epochs: int = 25,
    learning_rate: float = 1e-3,
    treatment: Optional[torch.Tensor] = None,
) -> None:
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
            loss = F.mse_loss(_predict(model, x, edge_index, device, treatment), y)
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
    treatment: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    predictions: List[np.ndarray] = []
    observations: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for window, target in zip(windows, targets):
            x = torch.tensor(window.astype(np.float32), device=device)
            pred = _predict(model, x, edge_index, device, treatment)
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
    model.eval()
    with torch.no_grad():
        x = torch.tensor(window.astype(np.float32), device=device)
        w = torch.tensor(treatment.astype(np.float32), device=device)
        parts = model.decompose(x, edge_index.to(device), treatment=w)
        effects = model.causal_effects(x, edge_index.to(device), treatment=w)

    return {
        "direct_effect_l1": float(parts.direct.abs().mean().item()),
        "spillover_effect_l1": float(parts.spillover.abs().mean().item()),
        "ite_l1": float(effects["ite"].abs().mean().item()),
        "ste_l1": float(effects["ste"].abs().mean().item()),
        "treated_fraction": float(parts.treatment.mean().item()),
        "mean_exposure": float(parts.exposure.mean().item()),
    }


def policy_boundary(n_neighborhoods: int, config: Optional[Dict[str, Any]] = None) -> List[int]:
    synth_cfg = (config or {}).get("synth", {})
    shock_nodes = synth_cfg.get("accessibility_shock_nodes", [0, 1, 4, 7])
    boundary = sorted({int(node) for node in shock_nodes if 0 <= int(node) < n_neighborhoods})
    return boundary or [0]


def treatment_vector(n_neighborhoods: int, boundary: Sequence[int]) -> np.ndarray:
    w = np.zeros(n_neighborhoods, dtype=np.float64)
    w[list(boundary)] = 1.0
    return w


def build_scenario_policies(
    n_neighborhoods: int, config: Optional[Dict[str, Any]] = None
) -> List[Policy]:
    boundary = policy_boundary(n_neighborhoods, config or {})
    policy_cfg = (config or {}).get("policy", {}) or {}

    def intensity(name: str, fallback: float) -> float:
        entry = policy_cfg.get(name, {}) or {}
        return float(entry.get("intensity", fallback))

    return [
        baseline_policy(),
        market_led_policy(boundary, timing=2, intensity=intensity("market_led", 0.8)),
        inclusionary_housing_policy(
            boundary, timing=1, intensity=intensity("inclusionary_housing", 0.6)
        ),
        land_value_capture_policy(
            boundary, timing=2, intensity=intensity("land_value_capture", 0.7)
        ),
    ]


def build_policy_scorecard(
    city: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    overburden_params: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    panel = city["panel"]
    config = city.get("config", {}) or {}
    n_neighborhoods = int(city.get("n_neighborhoods", panel.shape[0]))
    policies = build_scenario_policies(n_neighborhoods, config)

    scoring = config.get("scoring", {}) or {}
    if weights is None:
        weights = dict(scoring.get("weights", DEFAULT_DPI_WEIGHTS))
    if overburden_params is None:
        overburden_params = dict(scoring.get("overburden", DEFAULT_HT_PARAMS))

    return compare_policies(panel, policies, weights=weights, overburden_params=overburden_params)


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def shuffled_edge_index(edge_index: torch.Tensor, num_nodes: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(num_nodes, generator=generator)
    return permutation[edge_index]


def build_model_suite(
    n_features: int, hidden_dim: int, window_size: int
) -> Dict[str, Callable[[], torch.nn.Module]]:
    return {
        "stgnn": lambda: SpatioTemporalGNN(
            input_dim=n_features, hidden_dim=hidden_dim, output_dim=n_features
        ),
        "mlp": lambda: MLPBaseline(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            output_dim=n_features,
            sequence_length=window_size,
        ),
        "slm": lambda: SpatialLagModel(
            input_dim=n_features, output_dim=n_features, sequence_length=window_size
        ),
        "persistence": lambda: PersistenceBaseline(output_dim=n_features),
    }


def prepare_splits(
    panel: np.ndarray, window_size: int, scale_features: bool
) -> Tuple[List, List, List, List, List, Optional[FeatureScaler]]:
    windows, targets = make_windows(panel, window_size=window_size)
    if len(windows) < 2:
        raise ValueError(
            f"window_size={window_size} yields {len(windows)} windows; at least 2 are required"
        )
    split = max(1, int(len(windows) * 0.7))
    train_windows, train_targets = windows[:split], targets[:split]
    test_windows, test_targets = windows[split:], targets[split:]
    if not test_windows:
        train_windows, train_targets = windows[:-1], targets[:-1]
        test_windows, test_targets = windows[-1:], targets[-1:]

    raw_test_windows = list(test_windows)
    scaler: Optional[FeatureScaler] = None
    if scale_features:
        scaler = FeatureScaler.fit_windows(train_windows, feature_names=RQ1_FEATURE_NAMES)
        train_windows = [scaler.transform(w) for w in train_windows]
        train_targets = [scaler.transform(t) for t in train_targets]
        test_windows = [scaler.transform(w) for w in test_windows]
        test_targets = [scaler.transform(t) for t in test_targets]

    return train_windows, train_targets, test_windows, test_targets, raw_test_windows, scaler


def run_forecast_experiment(
    seed: int = 42,
    n_neighborhoods: int = 16,
    n_steps: int = 18,
    epochs: int = 300,
    config: Optional[Dict[str, Any]] = None,
    window_size: int = 4,
    hidden_dim: int = 12,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
    scale_features: bool = True,
    scenario: str = "baseline",
    topology: str = "real",
    model_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    torch_device = resolve_device(device)
    torch.manual_seed(seed)

    city = generate_synthetic_city(
        seed=seed, n_neighborhoods=n_neighborhoods, n_steps=n_steps, config=config
    )
    full_panel = city["panel"]
    policies = {p.name: p for p in build_scenario_policies(full_panel.shape[0], city["config"])}
    if scenario not in policies:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {sorted(policies)}")
    scenario_panel = apply_policy(full_panel, policies[scenario])

    panel = scenario_panel[:, :, RQ1_FEATURE_COLUMNS]
    n_features = len(RQ1_FEATURE_NAMES)
    real_edges = torch.tensor(city["edge_index"], dtype=torch.long)

    if topology == "real":
        edge_index = real_edges
    elif topology == "no_edges":
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    elif topology == "shuffled":
        edge_index = shuffled_edge_index(real_edges, panel.shape[0], seed)
    else:
        raise ValueError(f"unknown topology {topology!r}")

    splits = prepare_splits(panel, window_size, scale_features)
    train_windows, train_targets, test_windows, test_targets, raw_test_windows, scaler = splits

    boundary = policy_boundary(panel.shape[0], city["config"])
    treatment_np = treatment_vector(panel.shape[0], boundary)
    treatment = torch.tensor(treatment_np, dtype=torch.float32, device=torch_device)
    if scenario == "baseline":
        treatment = torch.zeros_like(treatment)

    row: Dict[str, float] = {
        "seed": int(seed),
        "scenario": scenario,
        "topology": topology,
        "n_neighborhoods": int(panel.shape[0]),
        "n_steps": int(n_steps),
    }
    trained: Dict[str, torch.nn.Module] = {}
    suite = build_model_suite(n_features, hidden_dim, window_size)
    if model_names is not None:
        unknown = [n for n in model_names if n not in suite]
        if unknown:
            raise ValueError(f"unknown model(s) {unknown}; expected from {sorted(suite)}")
        suite = {n: suite[n] for n in model_names}
    for name, build in suite.items():
        torch.manual_seed(seed)
        model = build().to(torch_device)
        trained[name] = model
        model_treatment = treatment if isinstance(model, SpatioTemporalGNN) else None
        train_model(
            model,
            train_windows,
            train_targets,
            edge_index,
            device=torch_device,
            epochs=epochs,
            learning_rate=learning_rate,
            treatment=model_treatment,
        )
        metrics = evaluate_model(
            model,
            test_windows,
            test_targets,
            edge_index,
            RQ1_FEATURE_NAMES,
            torch_device,
            scaler=scaler,
            treatment=model_treatment,
        )
        for key, value in metrics.items():
            row[f"{name}_{key}"] = value

    if "stgnn" in row_models(row):
        row["graph_overall_mae"] = row["stgnn_overall_mae"]
        row["graph_overall_rmse"] = row["stgnn_overall_rmse"]
    if "mlp" in row_models(row):
        row["baseline_overall_mae"] = row["mlp_overall_mae"]
        row["baseline_overall_rmse"] = row["mlp_overall_rmse"]
    if "stgnn" in trained and "mlp" in trained:
        row["graph_beats_baseline"] = float(row["stgnn_overall_mae"] < row["mlp_overall_mae"])

    if "stgnn" in trained:
        row.update(
            summarize_causal_split(
                trained["stgnn"], test_windows[-1], edge_index, treatment_np, torch_device
            )
        )
    return row


def row_models(row: Dict[str, float]) -> set:
    return {key.rsplit("_overall_mae", 1)[0] for key in row if key.endswith("_overall_mae")}


def run_multi_seed_benchmark(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(seeds) == 0:
        raise ValueError("at least one seed is required")

    rows = [run_forecast_experiment(seed=int(seed), **kwargs) for seed in seeds]
    per_seed = pd.DataFrame(rows)

    numeric = per_seed.select_dtypes(include=[np.number])
    metric_columns = [c for c in numeric.columns if c != "seed"]
    summary = pd.DataFrame(
        {
            "metric": metric_columns,
            "mean": [per_seed[c].mean() for c in metric_columns],
            "std": [per_seed[c].std(ddof=1) for c in metric_columns],
            "min": [per_seed[c].min() for c in metric_columns],
            "max": [per_seed[c].max() for c in metric_columns],
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
        description="RQ1 multi-seed benchmark: Causal ST-GNN vs. MLP, SLM and persistence."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-neighborhoods", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--scenario", default="baseline", choices=list(SCENARIOS))
    parser.add_argument("--topology", default="real", choices=["real", "no_edges", "shuffled"])
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--output", default="artifacts/rq1.csv")
    parser.add_argument("--summary-output", default="artifacts/rq1_summary.csv")
    parser.add_argument("--no-strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    device = resolve_device(args.device)
    print(f"RQ1 benchmark on {device} over seeds {args.seeds} [{args.scenario}/{args.topology}]")

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
        scenario=args.scenario,
        topology=args.topology,
    )

    print("\nPer-seed overall MAE:")
    for _, row in per_seed.iterrows():
        print(
            f"  seed={int(row['seed'])}: stgnn={row['stgnn_overall_mae']:.4f}, "
            f"mlp={row['mlp_overall_mae']:.4f}, slm={row['slm_overall_mae']:.4f}, "
            f"persistence={row['persistence_overall_mae']:.4f}"
        )

    print(f"\nAggregate over {len(per_seed)} seeds (mean +/- std):")
    indexed = summary.set_index("metric")
    for model in ("stgnn", "mlp", "slm", "persistence"):
        for metric in ("overall_mae", "overall_rmse"):
            key = f"{model}_{metric}"
            print(f"  {key}: {indexed.loc[key, 'mean']:.4f} +/- {indexed.loc[key, 'std']:.4f}")

    print("\nCausal channel magnitudes (ST-GNN):")
    for key in ("ite_l1", "ste_l1", "mean_exposure"):
        print(f"  {key}: {indexed.loc[key, 'mean']:.4f} +/- {indexed.loc[key, 'std']:.4f}")

    scorecard = build_policy_scorecard(
        generate_synthetic_city(
            seed=int(args.seeds[0]),
            n_neighborhoods=args.n_neighborhoods,
            n_steps=args.n_steps,
        )
    )
    print(f"\nPolicy scorecard (seed {args.seeds[0]}):")
    print(scorecard.to_string(index=False))

    _write_csv(per_seed, args.output)
    _write_csv(summary, args.summary_output)

    graph_mae = float(per_seed["stgnn_overall_mae"].mean())
    mlp_mae = float(per_seed["mlp_overall_mae"].mean())
    wins = int(per_seed["graph_beats_baseline"].sum())
    verdict = "PASS" if graph_mae < mlp_mae else "FAIL"
    print(
        f"\nRQ1 verdict: {verdict} -- ST-GNN mean MAE {graph_mae:.4f} vs. MLP {mlp_mae:.4f} "
        f"({wins}/{len(per_seed)} seeds won)"
    )

    if verdict == "FAIL" and not args.no_strict:
        raise RuntimeError(
            "RQ1 failed: the ST-GNN did not outperform the non-graph baseline "
            "on the synthetic spillover task (mean over seeds)."
        )


if __name__ == "__main__":
    main()
