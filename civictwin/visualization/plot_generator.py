from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from civictwin.evaluate import displacement_pressure_index, housing_transport_index
from civictwin.experiments.run_rq1 import (
    RQ1_FEATURE_COLUMNS,
    RQ1_FEATURE_NAMES,
    build_model_suite,
    policy_boundary,
    prepare_splits,
    train_model,
    treatment_vector,
)
from civictwin.synth import scenario_panels

FIGURE_DPI = 300
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "market_led": "Market-Led Development",
    "inclusionary_housing": "Inclusionary Housing",
    "land_value_capture": "Land-Value Capture",
}
SCENARIO_COLORS = {
    "baseline": "#4C4C4C",
    "market_led": "#C44E52",
    "inclusionary_housing": "#4C72B0",
    "land_value_capture": "#55A868",
}
MODEL_ORDER = ["stgnn", "slm", "mlp", "persistence"]
MODEL_LABELS = {
    "stgnn": "Causal ST-GNN",
    "slm": "Spatial Lag",
    "mlp": "Temporal MLP",
    "persistence": "Persistence",
}
MODEL_COLORS = {
    "stgnn": "#4C72B0",
    "slm": "#55A868",
    "mlp": "#C44E52",
    "persistence": "#8172B2",
}
TOPOLOGY_LABELS = {
    "real": "Real Graph",
    "no_edges": "No Edges",
    "shuffled": "Shuffled Edges",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": FIGURE_DPI,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=FIGURE_DPI)
        written.append(path)
    plt.close(fig)
    return written


def lattice_shape(n_nodes: int) -> Tuple[int, int]:
    rows = max(1, int(np.ceil(np.sqrt(n_nodes))))
    cols = max(1, int(np.ceil(n_nodes / rows)))
    if rows * cols < n_nodes:
        rows += 1
    return rows, cols


def to_grid(values: np.ndarray, n_nodes: int) -> np.ndarray:
    rows, cols = lattice_shape(n_nodes)
    grid = np.full(rows * cols, np.nan)
    grid[:n_nodes] = values[:n_nodes]
    return grid.reshape(rows, cols)


def hop_distances(adjacency: np.ndarray, sources: Sequence[int]) -> np.ndarray:
    n = adjacency.shape[0]
    distance = np.full(n, np.inf)
    frontier = list(sources)
    for node in frontier:
        distance[node] = 0
    current = 0
    while frontier:
        nxt = []
        for node in frontier:
            for neighbor in np.nonzero(adjacency[node])[0]:
                if distance[neighbor] == np.inf:
                    distance[neighbor] = current + 1
                    nxt.append(int(neighbor))
        frontier = nxt
        current += 1
    return distance


def train_reference_model(
    seed: int = 1,
    n_neighborhoods: int = 16,
    n_steps: int = 18,
    epochs: int = 250,
    window_size: int = 4,
    hidden_dim: int = 12,
) -> Dict[str, Any]:
    city = scenario_panels(seed=seed, n_neighborhoods=n_neighborhoods, n_steps=n_steps)
    panel = city["scenarios"]["market_led"][:, :, RQ1_FEATURE_COLUMNS]
    edge_index = torch.tensor(city["edge_index"], dtype=torch.long)
    device = torch.device("cpu")

    splits = prepare_splits(panel, window_size, True)
    train_windows, train_targets, test_windows, _, _, scaler = splits

    boundary = policy_boundary(panel.shape[0], city["config"])
    treatment_np = treatment_vector(panel.shape[0], boundary)
    treatment = torch.tensor(treatment_np, dtype=torch.float32)

    torch.manual_seed(seed)
    model = build_model_suite(len(RQ1_FEATURE_NAMES), hidden_dim, window_size)["stgnn"]()
    train_model(
        model,
        train_windows,
        train_targets,
        edge_index,
        device=device,
        epochs=epochs,
        treatment=treatment,
    )

    direct_series, spillover_series = [], []
    model.eval()
    with torch.no_grad():
        for window in test_windows:
            x = torch.tensor(window.astype(np.float32))
            effects = model.causal_effects(x, edge_index, treatment=treatment)
            direct_series.append(effects["ite"].abs().mean(dim=-1).numpy())
            spillover_series.append(effects["ste"].abs().mean(dim=-1).numpy())

    return {
        "city": city,
        "adjacency": city["adjacency"],
        "boundary": boundary,
        "treatment": treatment_np,
        "direct": np.stack(direct_series),
        "spillover": np.stack(spillover_series),
        "n_nodes": int(panel.shape[0]),
    }


def figure_spillover_heatmaps(output_dir: Path, reference: Dict[str, Any]) -> List[Path]:
    n_nodes = reference["n_nodes"]
    direct = reference["direct"]
    spillover = reference["spillover"]
    adjacency = reference["adjacency"]
    boundary = reference["boundary"]

    n_periods = min(3, direct.shape[0])
    period_idx = np.linspace(0, direct.shape[0] - 1, n_periods).astype(int)

    fig = plt.figure(figsize=(9.0, 5.4))
    grid = fig.add_gridspec(2, n_periods + 1, width_ratios=[1] * n_periods + [1.15])

    treatment_grid = to_grid(reference["treatment"], n_nodes)
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(treatment_grid, cmap="Greys", vmin=0, vmax=1)
    ax.set_title("Treated boundary $w_i$")
    ax.set_xticks([])
    ax.set_yticks([])
    for (r, c), value in np.ndenumerate(treatment_grid):
        if np.isfinite(value) and value > 0:
            ax.text(c, r, "T", ha="center", va="center", color="white", fontsize=8)

    direct_max = float(np.nanmax(direct)) or 1.0
    spill_max = float(np.nanmax(spillover)) or 1.0

    for col, t in enumerate(period_idx):
        if col == 0:
            continue
        ax = fig.add_subplot(grid[0, col])
        im = ax.imshow(to_grid(direct[t], n_nodes), cmap="Reds", vmin=0, vmax=direct_max)
        ax.set_title(f"Direct effect $\\Phi$, $t={t}$")
        ax.set_xticks([])
        ax.set_yticks([])
    cax = fig.add_subplot(grid[0, -1])
    fig.colorbar(im, cax=cax, label="|ITE|")

    for col, t in enumerate(period_idx):
        ax = fig.add_subplot(grid[1, col])
        im2 = ax.imshow(to_grid(spillover[t], n_nodes), cmap="Blues", vmin=0, vmax=spill_max)
        ax.set_title(f"Spillover $\\Psi$, $t={t}$")
        ax.set_xticks([])
        ax.set_yticks([])
    cax2 = fig.add_subplot(grid[1, -1])
    fig.colorbar(im2, cax=cax2, label="|STE|")

    distances = hop_distances(adjacency, boundary)
    fig.suptitle(
        "Direct policy impact vs. multi-hop neighbour spillover (Market-Led scenario)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    hop_text = []
    mean_spill = spillover.mean(axis=0)
    for hop in range(0, 4):
        mask = distances == hop if hop < 3 else distances >= 3
        if mask.any():
            hop_text.append(f"hop {hop}{'+' if hop == 3 else ''}: {mean_spill[mask].mean():.4f}")
    fig.text(
        0.5,
        0.015,
        "mean |STE| by graph distance from treated set   " + "   |   ".join(hop_text),
        ha="center",
        fontsize=8,
    )
    return save_figure(fig, output_dir, "spillover_heatmaps")


FLOORS = {0: 5.0, 1: 1.0, 2: 100.0, 3: 1000.0, 4: 50.0}


def degenerate_onset(panel: np.ndarray, tolerance: float = 1e-9) -> Optional[int]:
    for t in range(panel.shape[1]):
        for feature, floor in FLOORS.items():
            if np.any(panel[:, t, feature] <= floor + tolerance):
                return t
    return None


def figure_scenario_trajectories(
    output_dir: Path, seed: int = 1, n_neighborhoods: int = 16, horizon: int = 101
) -> List[Path]:
    city = scenario_panels(seed=seed, n_neighborhoods=n_neighborhoods, n_steps=horizon)
    baseline_panel = city["scenarios"]["baseline"]
    onset = degenerate_onset(baseline_panel)

    base_ht = housing_transport_index(baseline_panel).mean(axis=0)
    base_dpi = displacement_pressure_index(baseline_panel, per_step=True).mean(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0))

    for scenario, panel in city["scenarios"].items():
        ht = housing_transport_index(panel).mean(axis=0)
        dpi = displacement_pressure_index(panel, per_step=True).mean(axis=0)
        style = dict(
            label=SCENARIO_LABELS[scenario],
            color=SCENARIO_COLORS[scenario],
            linewidth=1.4,
        )
        axes[0, 0].plot(np.arange(ht.size), ht, **style)
        axes[1, 0].plot(np.arange(1, dpi.size + 1), dpi, **style)
        if scenario != "baseline":
            axes[0, 1].plot(np.arange(ht.size), ht - base_ht, **style)
            axes[1, 1].plot(np.arange(1, dpi.size + 1), dpi - base_dpi, **style)

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("H+T index (log)")
    axes[0, 0].set_title("(a) H+T level")
    axes[0, 1].set_yscale("symlog", linthresh=1e-4)
    axes[0, 1].set_ylabel("$\\Delta$ H+T vs. baseline")
    axes[0, 1].set_title("(b) H+T policy effect")
    axes[1, 0].set_ylabel("DPI")
    axes[1, 0].set_title("(c) DPI level")
    axes[1, 1].set_ylabel("$\\Delta$ DPI vs. baseline")
    axes[1, 1].set_title("(d) DPI policy effect")

    for ax in axes.ravel():
        ax.set_xlabel("simulation step $t$")
        if onset is not None and onset < horizon:
            ax.axvspan(onset, horizon, color="#B00020", alpha=0.10, zorder=0)
            ax.axvline(onset, color="#B00020", linewidth=0.9, linestyle="--")
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")

    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=7.5)

    handles = [
        Line2D([0], [0], color="#B00020", linestyle="--", linewidth=0.9),
        Line2D([0], [0], color="#B00020", alpha=0.25, linewidth=8),
    ]
    axes[0, 1].legend(
        handles,
        [f"floor onset $t={onset}$" if onset is not None else "no floor onset", "degenerate regime"],
        frameon=False,
        fontsize=7,
        loc="upper left",
    )

    fig.suptitle(
        f"Scenario trajectories over $t=0..{horizon - 1}$ (city-mean, seed {seed})", fontsize=11
    )
    caption = (
        "Panels (b) and (d) show each intervention relative to the untreated baseline, the "
        "comparative quantity of interest. Shaded region: at least one node has hit a hard "
        "state floor in synth.py (income clamps at 1000), so trajectories beyond it reflect "
        "the clamp rather than the generative dynamics and must not be interpreted."
    )
    fig.text(0.5, -0.035, caption, ha="center", fontsize=7.2, style="italic", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, output_dir, "scenario_trajectories")


def figure_model_comparison(
    output_dir: Path, summary: pd.DataFrame, scorecard: Optional[pd.DataFrame]
) -> List[Path]:
    real = summary[summary["topology"] == "real"]
    pooled = (
        real.groupby("model")
        .agg(
            mae=("overall_mae_mean", "mean"),
            mae_std=("overall_mae_std", "mean"),
            rmse=("overall_rmse_mean", "mean"),
            rmse_std=("overall_rmse_std", "mean"),
        )
        .reindex([m for m in MODEL_ORDER if m in real["model"].unique()])
    )

    has_dpi = scorecard is not None and not scorecard.empty
    n_panels = 3 if has_dpi else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.3))

    x = np.arange(len(pooled))
    colors = [MODEL_COLORS[m] for m in pooled.index]
    labels = [MODEL_LABELS[m] for m in pooled.index]

    for ax, metric, err, title in (
        (axes[0], "mae", "mae_std", "Forecast MAE"),
        (axes[1], "rmse", "rmse_std", "Forecast RMSE"),
    ):
        ax.bar(x, pooled[metric], yerr=pooled[err], color=colors, capsize=3, width=0.66)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(metric.upper())

    if has_dpi:
        ax = axes[2]
        order = scorecard.sort_values("pressure_delta_mean")
        pos = np.arange(len(order))
        ax.barh(
            pos,
            order["pressure_delta_mean"],
            xerr=order["pressure_delta_std"].fillna(0.0),
            color=[SCENARIO_COLORS.get(p, "#888888") for p in order["policy"]],
            capsize=3,
        )
        ax.set_yticks(pos)
        ax.set_yticklabels([SCENARIO_LABELS.get(p, p) for p in order["policy"]], fontsize=7.5)
        ax.axvline(0.0, color="black", linewidth=0.6)
        ax.set_title("DPI change vs. baseline")
        ax.set_xlabel("$\\Delta$ DPI")

    fig.suptitle("Model family comparison and policy DPI mitigation", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_figure(fig, output_dir, "model_comparison_bar")


def figure_ablation_topology(output_dir: Path, ablation: pd.DataFrame) -> List[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    pooled = (
        ablation.groupby("topology")
        .agg(mae=("overall_mae_mean", "mean"), std=("overall_mae_std", "mean"))
        .reindex([t for t in TOPOLOGY_LABELS if t in ablation["topology"].unique()])
    )
    x = np.arange(len(pooled))
    axes[0].bar(
        x,
        pooled["mae"],
        yerr=pooled["std"],
        color=["#4C72B0", "#C44E52", "#DD8452"][: len(pooled)],
        capsize=3,
        width=0.6,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([TOPOLOGY_LABELS[t] for t in pooled.index], rotation=12)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("ST-GNN accuracy by graph topology")

    scenarios = list(ablation["scenario"].unique())
    width = 0.26
    base = np.arange(len(scenarios))
    for offset, topology in enumerate(
        [t for t in TOPOLOGY_LABELS if t in ablation["topology"].unique()]
    ):
        subset = ablation[ablation["topology"] == topology].set_index("scenario")
        values = [subset.loc[s, "overall_mae_mean"] if s in subset.index else np.nan for s in scenarios]
        errors = [subset.loc[s, "overall_mae_std"] if s in subset.index else 0.0 for s in scenarios]
        axes[1].bar(
            base + (offset - 1) * width,
            values,
            width=width,
            yerr=errors,
            capsize=2,
            label=TOPOLOGY_LABELS[topology],
        )
    axes[1].set_xticks(base)
    axes[1].set_xticklabels(
        [SCENARIO_LABELS.get(s, s) for s in scenarios], rotation=18, ha="right", fontsize=7.5
    )
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Degradation per scenario")
    axes[1].legend(frameon=False, fontsize=7.5)

    fig.suptitle("Graph topology ablation: real vs. removed vs. randomised edges", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_figure(fig, output_dir, "ablation_topology")


def generate_all_figures(
    results_dir: str | Path = "./results",
    output_dir: str | Path = "./paper_assets/figures",
    seed: int = 1,
    horizon: int = 101,
    epochs: int = 250,
) -> List[Path]:
    apply_style()
    results = Path(results_dir)
    figures = Path(output_dir)
    written: List[Path] = []

    reference = train_reference_model(seed=seed, epochs=epochs)
    written += figure_spillover_heatmaps(figures, reference)
    written += figure_scenario_trajectories(figures, seed=seed, horizon=horizon)

    summary_path = results / "benchmark_summary.csv"
    scorecard_path = results / "policy_scorecard.csv"
    ablation_path = results / "ablation_topology.csv"

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        scorecard = pd.read_csv(scorecard_path) if scorecard_path.exists() else None
        written += figure_model_comparison(figures, summary, scorecard)
    else:
        print(f"skipping model_comparison_bar: {summary_path} not found")

    if ablation_path.exists():
        written += figure_ablation_topology(figures, pd.read_csv(ablation_path))
    else:
        print(f"skipping ablation_topology: {ablation_path} not found")

    for path in written:
        print(f"wrote {path}")
    return written


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CG-CEN publication figures.")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output-dir", default="./paper_assets/figures")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=101)
    parser.add_argument("--epochs", type=int, default=250)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    generate_all_figures(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        horizon=args.horizon,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
