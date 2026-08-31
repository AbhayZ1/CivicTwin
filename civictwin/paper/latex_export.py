from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

MODEL_ORDER = ["stgnn", "slm", "mlp", "persistence"]
MODEL_LABELS = {
    "stgnn": r"Causal ST-GNN (CG-CEN)",
    "slm": r"Spatial Lag Model",
    "mlp": r"Temporal MLP",
    "persistence": r"Naive Persistence",
}
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "market_led": "Market-Led Development",
    "inclusionary_housing": "Inclusionary Housing",
    "land_value_capture": "Land-Value Capture",
}


def escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = str(text)
    for key, value in replacements.items():
        out = out.replace(key, value)
    return out


def format_p_value(value: float) -> str:
    if value is None or not np.isfinite(value):
        return r"--"
    if value < 0.001:
        return r"$<0.001$"
    return f"${value:.3f}$"


def significance_marker(value: float) -> str:
    if value is None or not np.isfinite(value):
        return ""
    if value < 0.001:
        return r"^{***}"
    if value < 0.01:
        return r"^{**}"
    if value < 0.05:
        return r"^{*}"
    return ""


def mean_std(mean: float, std: float, decimals: int = 3, marker: str = "") -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return f"${mean:.{decimals}f}{marker}$"
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}{marker}$"


def build_benchmark_table(
    summary: pd.DataFrame,
    significance: Optional[pd.DataFrame] = None,
    topology: str = "real",
) -> str:
    real = summary[summary["topology"] == topology]
    scenarios = [s for s in SCENARIO_LABELS if s in list(real["scenario"].unique())]
    models = [m for m in MODEL_ORDER if m in list(real["model"].unique())]

    pvalues: Dict[str, Dict[str, float]] = {}
    if significance is not None and not significance.empty:
        for _, row in significance.iterrows():
            comparison = str(row["comparison"])
            if not comparison.startswith("stgnn_vs_"):
                continue
            opponent = comparison.split("stgnn_vs_", 1)[1]
            pvalues.setdefault(str(row["scenario"]), {})[opponent] = float(row["p_value"])

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Forecast accuracy across policy scenarios on the synthetic CivicTwin "
        r"panel. Values are mean $\pm$ standard deviation of the pooled test-set error over "
        r"independent seeds. Significance markers denote a paired $t$-test of each baseline "
        r"against the Causal ST-GNN on MAE "
        r"($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$).}"
    )
    lines.append(r"\label{tab:benchmark_results}")
    lines.append(r"\small")
    column_spec = "l" + "cc" * len(scenarios)
    lines.append(rf"\begin{{tabular}}{{{column_spec}}}")
    lines.append(r"\toprule")

    header = [r"\textbf{Model}"]
    for scenario in scenarios:
        header.append(
            r"\multicolumn{2}{c}{\textbf{" + escape(SCENARIO_LABELS[scenario]) + r"}}"
        )
    lines.append(" & ".join(header) + r" \\")

    cmid = []
    for idx in range(len(scenarios)):
        start = 2 + idx * 2
        cmid.append(rf"\cmidrule(lr){{{start}-{start + 1}}}")
    lines.append("".join(cmid))
    lines.append(" & " + " & ".join([r"MAE & RMSE"] * len(scenarios)) + r" \\")
    lines.append(r"\midrule")

    for model in models:
        cells = [MODEL_LABELS.get(model, escape(model))]
        for scenario in scenarios:
            row = real[(real["scenario"] == scenario) & (real["model"] == model)]
            if row.empty:
                cells += ["--", "--"]
                continue
            record = row.iloc[0]
            marker = ""
            if model != "stgnn":
                marker = significance_marker(pvalues.get(scenario, {}).get(model, float("nan")))
            cells.append(
                mean_std(record["overall_mae_mean"], record["overall_mae_std"], marker=marker)
            )
            cells.append(mean_std(record["overall_rmse_mean"], record["overall_rmse_std"]))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    prow = [r"\textit{p} vs. ST-GNN (MAE)"]
    for scenario in scenarios:
        best = min(
            (pvalues.get(scenario, {}).get(m, float("nan")) for m in models if m != "stgnn"),
            default=float("nan"),
            key=lambda v: v if np.isfinite(v) else np.inf,
        )
        prow.append(r"\multicolumn{2}{c}{" + format_p_value(best) + r"}")
    lines.append(" & ".join(prow) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def build_scorecard_table(scorecard: pd.DataFrame) -> str:
    order = scorecard.sort_values("pressure_delta_mean").reset_index(drop=True)

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Policy scenario scorecard ranked by displacement-pressure mitigation. "
        r"Negative values indicate that the intervention reduces the index relative to the "
        r"untreated baseline panel. Values are means over independent seeds.}"
    )
    lines.append(r"\label{tab:policy_scorecard}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{clccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Rank} & \textbf{Scenario} & $\Delta$\textbf{DPI} & "
        r"$\Delta$\textbf{H+T} & $\Delta$\textbf{Rent/Income} \\"
    )
    lines.append(r"\midrule")

    for rank, row in order.iterrows():
        lines.append(
            " & ".join(
                [
                    str(int(rank) + 1),
                    escape(SCENARIO_LABELS.get(row["policy"], row["policy"])),
                    mean_std(row["pressure_delta_mean"], row.get("pressure_delta_std", np.nan), 5),
                    mean_std(
                        row["overburden_delta_mean"], row.get("overburden_delta_std", np.nan), 5
                    ),
                    f"${row['affordability_delta_mean']:.3e}$",
                ]
            )
            + r" \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_algorithm_block() -> str:
    return r"""\begin{algorithm}[t]
\caption{Causal ST-GNN forward pass with direct/spillover decomposition (CG-CEN)}
\label{alg:causal_stgnn}
\begin{algorithmic}[1]
\REQUIRE Node history $X_{i,t-W+1:t} \in \mathbb{R}^{W \times F}$ for all $i \in \mathcal{V}$;
         policy assignment $P_{i,t} \in \{0,1\}$; adjacency $\mathcal{N}(\cdot)$ with
         row-normalised weights $W_{ij}$
\ENSURE  Prediction $\hat{Y}_{i,t+1}$, individual treatment effect $\tau_i$,
         spatial interference effect $\eta_i$
\STATE $\mu, \sigma \gets$ \textsc{FitScaler}$(X_{\text{train}})$ \COMMENT{training split only}
\STATE $Z_{i,\cdot} \gets (X_{i,\cdot} - \mu) / \sigma$ for all $i$
\FOR{$k = t-W+1$ \TO $t$}
    \STATE $e_{i,k} \gets \mathrm{ReLU}(\mathbf{W}_{\text{enc}} Z_{i,k} + b_{\text{enc}})$
\ENDFOR
\STATE $h_i \gets \mathrm{GRU}\big(e_{i,t-W+1}, \dots, e_{i,t}\big)$ \COMMENT{node-local; no message passing}
\STATE $s_i \gets [\,h_i \,\Vert\, P_{i,t}\,]$
\STATE $\Phi_i \gets \mathrm{MLP}_{\Phi}(s_i) + X_{i,t}$ \COMMENT{direct term, residual anchor}
\STATE $\Psi_j \gets \mathrm{MLP}_{\Psi}(s_j)$ for all $j$
\STATE $\Omega_i \gets \sum_{j \in \mathcal{N}(i)} W_{ij} \, \Psi_j$ \COMMENT{spatial interference}
\STATE $\hat{Y}_{i,t+1} \gets \Phi_i + \Omega_i$
\STATE $s_i^{0} \gets [\,h_i \,\Vert\, 0\,]$ \COMMENT{counterfactual: no intervention}
\STATE $\tau_i \gets \mathrm{MLP}_{\Phi}(s_i) - \mathrm{MLP}_{\Phi}(s_i^{0})$ \COMMENT{ITE}
\STATE $\eta_i \gets \sum_{j \in \mathcal{N}(i)} W_{ij}\big(\mathrm{MLP}_{\Psi}(s_j)
        - \mathrm{MLP}_{\Psi}(s_j^{0})\big)$ \COMMENT{STE}
\RETURN $\hat{Y}_{i,t+1}, \tau_i, \eta_i$
\end{algorithmic}
\end{algorithm}
"""


def generate_all_latex(
    results_dir: str | Path = "./results",
    output_dir: str | Path = "./paper_assets",
) -> List[Path]:
    results = Path(results_dir)
    tables_dir = Path(output_dir) / "tables"
    algorithms_dir = Path(output_dir) / "algorithms"
    tables_dir.mkdir(parents=True, exist_ok=True)
    algorithms_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []

    summary_path = results / "benchmark_summary.csv"
    significance_path = results / "benchmark_significance.csv"
    scorecard_path = results / "policy_scorecard.csv"

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        significance = (
            pd.read_csv(significance_path) if significance_path.exists() else None
        )
        path = tables_dir / "table_benchmark_results.tex"
        path.write_text(build_benchmark_table(summary, significance), encoding="utf-8")
        written.append(path)
    else:
        print(f"skipping benchmark table: {summary_path} not found")

    if scorecard_path.exists():
        path = tables_dir / "table_policy_scorecard.tex"
        path.write_text(build_scorecard_table(pd.read_csv(scorecard_path)), encoding="utf-8")
        written.append(path)
    else:
        print(f"skipping scorecard table: {scorecard_path} not found")

    algo_path = algorithms_dir / "algo_causal_stgnn.tex"
    algo_path.write_text(build_algorithm_block(), encoding="utf-8")
    written.append(algo_path)

    for path in written:
        print(f"wrote {path}")
    return written


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CG-CEN LaTeX tables and algorithms.")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output-dir", default="./paper_assets")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    generate_all_latex(results_dir=args.results_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
