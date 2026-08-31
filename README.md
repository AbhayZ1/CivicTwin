# CivicTwin

CivicTwin is a lightweight synthetic research prototype for comparing urban policy scenarios using a spatio-temporal graph model. The current iteration intentionally avoids real-world data and instead uses a deterministic synthetic city with known dynamics so the pipeline can be validated in isolation.

## Goals

- Model neighborhood interactions over time with a graph-aware architecture
- Compare synthetic policy scenarios against a no-policy baseline
- Score policies on affordability and a transparent displacement-pressure index
- Validate the basic pipeline before introducing real data

## Structure

- `civictwin/synth.py`: synthetic city and graph generation
- `civictwin/graph.py`: graph assembly, including a no-edge ablation
- `civictwin/policy.py`: policy definitions and intervention logic
- `civictwin/model.py`: Causal ST-GNN with explicit Phi/Psi decomposition, plus MLP, Spatial Lag and persistence baselines
- `civictwin/scaling.py`: z-score feature normalisation fit on the training split only
- `civictwin/evaluate.py`: H+T index, DPI, MAE and true RMSE
- `civictwin/data/empirical_loader.py`: NYC tract/PLUTO pipeline to a PyTorch Geometric graph
- `civictwin/experiments/run_rq1.py`: single-scenario RQ1 experiment
- `civictwin/experiments/run_benchmark.py`: multi-seed, multi-scenario, multi-topology benchmark
- `civictwin/visualization/plot_generator.py`: publication figures (300 DPI PDF + PNG)
- `civictwin/paper/latex_export.py`: booktabs tables and the IEEE algorithm block
- `configs/default.yaml`: all synthetic and scoring defaults
- `tests/`: deterministic regression tests for the pipeline

## Quick start

1. Create a virtual environment and install dependencies.
2. Run the test suite:
   ```bash
   python -m pytest -q
   ```
3. Run the RQ1 experiment:
   ```bash
   python -m civictwin.experiments.run_rq1
   ```
   The default sweep runs five seeds and reports mean +/- std. To change the
   seed set, target a GPU, or redirect the metric CSVs:
   ```bash
   python -m civictwin.experiments.run_rq1 --seeds 1 2 3 4 5 --device cuda        --output artifacts/rq1.csv --summary-output artifacts/rq1_summary.csv
   ```
   `run_rq1` exits non-zero when the graph model loses to the non-graph baseline
   on mean MAE; pass `--no-strict` (or use `run_benchmark`) to report the verdict
   instead. The runner also prints a policy scorecard for the baseline,
   market-led, inclusionary-housing, and land-value-capture scenarios.

## Causal decomposition

Under partial interference, a neighbourhood outcome depends on its own treatment
`w_i` and on its neighbours' treatments through a degree-normalised exposure
`e_i = (sum_j A_ij w_j) / max(deg_i, 1)`. `SpatioTemporalGNN` predicts

```
Y_hat_i = mu_i + tau_i * w_i + gamma_i * e_i
```

where `mu` is the untreated counterfactual, `tau` is the **direct** policy effect
and `gamma` is the **spatial interference** (spillover) effect. The split is
enforced architecturally: `tau` reads a node-local temporal stream that never
touches a message-passing operator, while `gamma` reads the message-passed
stream. Call `model.decompose(x, edge_index, treatment)` to recover the isolated
components, or `direct_effect` / `interference_effect` for one channel.

With `w = 0` the model reduces exactly to `mu`; with no edges `e = 0` and the
interference term vanishes, so the no-edge ablation is a strict special case.

## Metrics

- **H+T Overburden Index** - combined housing and transportation cost share of
  income, minus the affordability threshold (CNT benchmark `0.45`), clipped at
  zero. Transport cost falls with accessibility, which is the mechanism a
  housing-only rent-to-income ratio misses. Parameters live under
  `scoring.overburden` in `configs/default.yaml`.
- **Displacement Pressure Index (DPI)** - a weighted blend of adverse
  period-over-period movements: rent growth, income decline and housing-stock
  contraction. Weights live under `scoring.weights`.
- **MAE / RMSE** - pooled over every window, node and feature, so RMSE is a true
  root mean squared error rather than a root-mean-square of per-window MAEs.

Both indices are constructed comparative indicators over a synthetic panel.

## Reproduction pipeline

```bash
python -m pytest -q

python -m civictwin.experiments.run_benchmark --seeds 10 --output_dir ./results
python -m civictwin.visualization.plot_generator --results-dir ./results
python -m civictwin.paper.latex_export --results-dir ./results
```

`run_benchmark` accepts `--seeds 10` (a count) or an explicit list (`--seeds 1 2 3`). It
exports the per-seed scenario datasets to `./data/synthetic/`, writes raw and aggregated
results to `./results/`, and reports mean, standard deviation, 95% confidence intervals and
paired t-tests of the Causal ST-GNN against each baseline.

Outputs:

- `results/benchmark_raw.csv`, `benchmark_tidy.csv`, `benchmark_summary.csv`
- `results/benchmark_significance.csv`, `ablation_topology.csv`, `policy_scorecard.csv`
- `paper_assets/figures/*.{pdf,png}`
- `paper_assets/tables/*.tex`, `paper_assets/algorithms/algo_causal_stgnn.tex`

## Causal decomposition

Under partial interference the one-step-ahead prediction factorises as

```
Y_hat_{i,t+1} = Phi(X_{i,t}, P_{i,t}) + sum_{j in N(i)} W_ij * Psi(X_{j,t}, P_{j,t})
```

`Phi` is the direct term and carries the Individual Treatment Effect; `Psi`, aggregated over
neighbours with row-normalised weights `W_ij`, carries the Spatial Interference Effect.
`model.causal_effects(...)` returns `ite`, `ste` and `total` by contrasting each term against
its `P = 0` counterfactual, so the ITE is exactly zero on untreated nodes and the STE is
exactly zero on an edgeless graph.

`Phi` is anchored on the last observation (residual forecasting). Without that anchor the GRU
must reconstruct the level through a tanh bottleneck, which costs roughly an order of
magnitude in MAE.

## Known limitations

- The generator applies multiplicative rent growth with no equilibrating supply response.
  State variables hit the hard floors in `synth.py` (income clamps at 1000) around `t = 52`
  at default settings; trajectories past that point reflect the clamp, not the dynamics.
  Benchmarks use `n_steps = 18`, well inside the stable regime.
- DPI is a growth-rate index, so a permanent policy level-shift appears only as a one-step
  transient rather than a sustained offset.
- `empirical_loader` ships a schema-compatible **sample** table, not real ACS/PLUTO data.
  Supply a real extract via `load_empirical_city(table_path=...)`; `is_synthetic_sample`
  reports which source was used.

## Synthetic assumptions

The synthetic generation is intentionally transparent and not meant to be a real policy model. It creates:

- accessibility shocks that affect specific neighborhoods
- land-value increases that spur rent growth with a lag
- spillover effects to nearby neighborhoods via graph adjacency
- income and housing changes that respond to price pressure

These dynamics are simple by design and are meant to validate the modeling pipeline, not to claim real-world validity.

## Important caveat

The displacement-pressure score is a constructed comparative indicator, not a validated prediction of actual displacement. It is a tool for comparing synthetic scenarios in a controlled setting.
