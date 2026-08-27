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
- `civictwin/model.py`: ST-GNN with an explicit direct/interference decomposition, plus MLP and persistence baselines
- `civictwin/scaling.py`: z-score feature normalisation fit on the training split only
- `civictwin/evaluate.py`: affordability, H+T overburden, displacement scoring and forecast metrics
- `civictwin/experiments/run_rq1.py`: RQ1 multi-seed benchmark
- `civictwin/experiments/run_benchmark.py`: benchmark entry point (non-strict alias of the above)
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

## Synthetic assumptions

The synthetic generation is intentionally transparent and not meant to be a real policy model. It creates:

- accessibility shocks that affect specific neighborhoods
- land-value increases that spur rent growth with a lag
- spillover effects to nearby neighborhoods via graph adjacency
- income and housing changes that respond to price pressure

These dynamics are simple by design and are meant to validate the modeling pipeline, not to claim real-world validity.

## Important caveat

The displacement-pressure score is a constructed comparative indicator, not a validated prediction of actual displacement. It is a tool for comparing synthetic scenarios in a controlled setting.
