from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from civictwin.experiments.run_rq1 import (
    SCENARIOS,
    build_policy_scorecard,
    run_forecast_experiment,
)
from civictwin.synth import export_scenario_datasets, generate_synthetic_city

MODEL_NAMES = ("stgnn", "mlp", "slm", "persistence")
MODEL_LABELS = {
    "stgnn": "Causal ST-GNN",
    "mlp": "Temporal MLP",
    "slm": "Spatial Lag Model",
    "persistence": "Naive Persistence",
}
TOPOLOGIES = ("real", "no_edges", "shuffled")
METRICS = ("overall_mae", "overall_rmse")
DEFAULT_N_SEEDS = 10


def resolve_seeds(seeds: Sequence[int]) -> List[int]:
    if len(seeds) == 1 and seeds[0] > 1:
        return list(range(1, int(seeds[0]) + 1))
    return [int(s) for s in seeds]


def confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    n = clean.size
    if n < 2:
        single = float(clean[0]) if n == 1 else float("nan")
        return single, single
    mean = float(clean.mean())
    half = float(
        stats.t.ppf(0.5 + confidence / 2.0, df=n - 1) * clean.std(ddof=1) / np.sqrt(n)
    )
    return mean - half, mean + half


def paired_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    if left.size < 2:
        return {"t_statistic": float("nan"), "p_value": float("nan"), "n_pairs": int(left.size)}
    result = stats.ttest_rel(left, right)
    difference = left - right
    pooled = difference.std(ddof=1)
    return {
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": int(left.size),
        "mean_difference": float(difference.mean()),
        "cohens_dz": float(difference.mean() / pooled) if pooled > 0 else float("nan"),
    }


def run_main_sweep(
    seeds: Sequence[int],
    scenarios: Sequence[str],
    **kwargs: Any,
) -> pd.DataFrame:
    rows = []
    total = len(seeds) * len(scenarios)
    done = 0
    for scenario in scenarios:
        for seed in seeds:
            rows.append(
                run_forecast_experiment(
                    seed=int(seed), scenario=scenario, topology="real", **kwargs
                )
            )
            done += 1
            print(f"  main   [{done}/{total}] scenario={scenario} seed={seed}", flush=True)
    return pd.DataFrame(rows)


def run_topology_ablation(
    seeds: Sequence[int],
    scenarios: Sequence[str],
    **kwargs: Any,
) -> pd.DataFrame:
    rows = []
    ablated = [t for t in TOPOLOGIES if t != "real"]
    total = len(seeds) * len(scenarios) * len(ablated)
    done = 0
    for topology in ablated:
        for scenario in scenarios:
            for seed in seeds:
                rows.append(
                    run_forecast_experiment(
                        seed=int(seed),
                        scenario=scenario,
                        topology=topology,
                        model_names=["stgnn"],
                        **kwargs,
                    )
                )
                done += 1
                print(
                    f"  ablate [{done}/{total}] topology={topology} "
                    f"scenario={scenario} seed={seed}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def tidy_results(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in frame.iterrows():
        for model in MODEL_NAMES:
            key = f"{model}_overall_mae"
            if key not in row or not np.isfinite(row.get(key, np.nan)):
                continue
            record = {
                "seed": int(row["seed"]),
                "scenario": row["scenario"],
                "topology": row["topology"],
                "model": model,
                "model_label": MODEL_LABELS[model],
            }
            for metric in METRICS:
                record[metric] = float(row[f"{model}_{metric}"])
            records.append(record)
    return pd.DataFrame(records)


def summarize(tidy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = tidy.groupby(["scenario", "topology", "model", "model_label"], sort=False)
    for (scenario, topology, model, label), group in grouped:
        record = {
            "scenario": scenario,
            "topology": topology,
            "model": model,
            "model_label": label,
            "n_seeds": int(len(group)),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            low, high = confidence_interval(values)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        rows.append(record)
    return pd.DataFrame(rows)


def significance_tests(tidy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    real = tidy[tidy["topology"] == "real"]
    for scenario in real["scenario"].unique():
        subset = real[real["scenario"] == scenario]
        pivot = subset.pivot_table(index="seed", columns="model", values="overall_mae")
        if "stgnn" not in pivot:
            continue
        for opponent in ("mlp", "slm", "persistence"):
            if opponent not in pivot:
                continue
            paired = pivot[["stgnn", opponent]].dropna()
            result = paired_test(
                paired["stgnn"].to_numpy(), paired[opponent].to_numpy()
            )
            rows.append(
                {
                    "scenario": scenario,
                    "comparison": f"stgnn_vs_{opponent}",
                    "metric": "overall_mae",
                    **result,
                }
            )
    return pd.DataFrame(rows)


def ablation_table(tidy: pd.DataFrame) -> pd.DataFrame:
    stgnn = tidy[tidy["model"] == "stgnn"]
    rows = []
    for scenario in stgnn["scenario"].unique():
        subset = stgnn[stgnn["scenario"] == scenario]
        pivot = subset.pivot_table(index="seed", columns="topology", values="overall_mae")
        if "real" not in pivot:
            continue
        for topology in TOPOLOGIES:
            if topology not in pivot:
                continue
            paired = pivot[["real", topology]].dropna()
            values = paired[topology].to_numpy()
            low, high = confidence_interval(values)
            record = {
                "scenario": scenario,
                "topology": topology,
                "overall_mae_mean": float(values.mean()),
                "overall_mae_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "overall_mae_ci_low": low,
                "overall_mae_ci_high": high,
                "degradation_vs_real": float(
                    values.mean() - paired["real"].to_numpy().mean()
                ),
            }
            if topology != "real":
                record.update(
                    {
                        f"vs_real_{k}": v
                        for k, v in paired_test(
                            paired[topology].to_numpy(), paired["real"].to_numpy()
                        ).items()
                    }
                )
            rows.append(record)
    return pd.DataFrame(rows)


def scenario_scorecard(seeds: Sequence[int], n_neighborhoods: int, n_steps: int) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        city = generate_synthetic_city(
            seed=int(seed), n_neighborhoods=n_neighborhoods, n_steps=n_steps
        )
        card = build_policy_scorecard(city)
        card["seed"] = int(seed)
        frames.append(card)
    combined = pd.concat(frames, ignore_index=True)
    aggregated = (
        combined.groupby("policy", sort=False)
        .agg(
            overburden_delta_mean=("overburden_delta", "mean"),
            overburden_delta_std=("overburden_delta", "std"),
            pressure_delta_mean=("pressure_delta", "mean"),
            pressure_delta_std=("pressure_delta", "std"),
            affordability_delta_mean=("affordability_delta", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    aggregated["ranking"] = aggregated["pressure_delta_mean"].rank(method="dense")
    return aggregated


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path}")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CG-CEN multi-seed benchmark across models, scenarios and topologies."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_N_SEEDS])
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--n-neighborhoods", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="./results")
    parser.add_argument("--export-data-dir", default="./data/synthetic")
    parser.add_argument("--no-export-data", action="store_true")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--no-scale", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    seeds = resolve_seeds(args.seeds)
    output_dir = Path(args.output_dir)

    print(f"CG-CEN benchmark: {len(seeds)} seeds x {len(args.scenarios)} scenarios")
    print(f"seeds={seeds}")
    print(f"scenarios={list(args.scenarios)}")

    shared = dict(
        n_neighborhoods=args.n_neighborhoods,
        n_steps=args.n_steps,
        epochs=args.epochs,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        device=args.device,
        scale_features=not args.no_scale,
    )

    if not args.no_export_data:
        manifest = export_scenario_datasets(
            seeds=seeds,
            n_neighborhoods=args.n_neighborhoods,
            n_steps=args.n_steps,
            output_dir=args.export_data_dir,
        )
        print(f"wrote {manifest}")

    main_frame = run_main_sweep(seeds, args.scenarios, **shared)
    frames = [main_frame]
    if not args.no_ablation:
        frames.append(run_topology_ablation(seeds, args.scenarios, **shared))
    raw = pd.concat(frames, ignore_index=True)

    tidy = tidy_results(raw)
    summary = summarize(tidy)
    stats_frame = significance_tests(tidy)
    ablation = ablation_table(tidy)
    scorecard = scenario_scorecard(seeds, args.n_neighborhoods, args.n_steps)

    _write(raw, output_dir / "benchmark_raw.csv")
    _write(tidy, output_dir / "benchmark_tidy.csv")
    _write(summary, output_dir / "benchmark_summary.csv")
    _write(stats_frame, output_dir / "benchmark_significance.csv")
    _write(ablation, output_dir / "ablation_topology.csv")
    _write(scorecard, output_dir / "policy_scorecard.csv")

    payload = {
        "seeds": seeds,
        "scenarios": list(args.scenarios),
        "topologies": list(TOPOLOGIES),
        "models": {k: MODEL_LABELS[k] for k in MODEL_NAMES},
        "config": shared,
        "summary": summary.to_dict(orient="records"),
        "significance": stats_frame.to_dict(orient="records"),
        "ablation": ablation.to_dict(orient="records"),
        "policy_scorecard": scorecard.to_dict(orient="records"),
    }
    json_path = output_dir / "benchmark_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")

    print("\nOverall MAE by model (real topology, pooled over scenarios):")
    pooled = (
        tidy[tidy["topology"] == "real"]
        .groupby("model_label")["overall_mae"]
        .agg(["mean", "std", "count"])
        .sort_values("mean")
    )
    print(pooled.to_string())

    if not stats_frame.empty:
        print("\nPaired t-tests (ST-GNN vs baselines, per scenario):")
        print(
            stats_frame[
                ["scenario", "comparison", "mean_difference", "t_statistic", "p_value"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
