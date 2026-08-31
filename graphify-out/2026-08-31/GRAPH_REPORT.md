# Graph Report - CivicTwin  (2026-08-31)

## Corpus Check
- 24 files · ~34,688 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 250 nodes · 631 edges · 15 communities (13 shown, 2 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `37835e21`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- run_rq1.py
- SpatioTemporalGNN
- run_forecast_experiment
- test_cgcen.py
- plot_generator.py
- run_benchmark.py
- latex_export.py
- load_empirical_city
- FeatureScaler
- experiments/__init__.py
- civictwin/__init__.py
- synth.py
- CivicTwin
- Project Status (last updated 2026-08-31, commit `feed8f4`+)
- build_graph

## God Nodes (most connected - your core abstractions)
1. `run_forecast_experiment()` - 23 edges
2. `SpatioTemporalGNN` - 22 edges
3. `generate_synthetic_city()` - 18 edges
4. `scenario_panels()` - 15 edges
5. `displacement_pressure_index()` - 14 edges
6. `compare_policies()` - 14 edges
7. `Policy` - 14 edges
8. `main()` - 13 edges
9. `load_empirical_city()` - 12 edges
10. `train_model()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `test_scenario_panels_cover_all_four_policies()` --calls--> `scenario_panels()`  [EXTRACTED]
  tests/test_cgcen.py → civictwin/synth.py
- `test_baseline_policy_is_noop_and_eval_works()` --calls--> `affordability()`  [EXTRACTED]
  tests/test_graph_policy_eval.py → civictwin/evaluate.py
- `test_baseline_policy_is_noop_and_eval_works()` --calls--> `displacement_pressure_index()`  [EXTRACTED]
  tests/test_graph_policy_eval.py → civictwin/evaluate.py
- `test_baseline_policy_is_noop_and_eval_works()` --calls--> `compare_policies()`  [EXTRACTED]
  tests/test_graph_policy_eval.py → civictwin/evaluate.py
- `test_policy_scorecard_includes_configured_scenarios()` --calls--> `build_policy_scorecard()`  [EXTRACTED]
  tests/test_rq1.py → civictwin/experiments/run_rq1.py

## Import Cycles
- None detected.

## Communities (15 total, 2 thin omitted)

### Community 0 - "run_rq1.py"
Cohesion: 0.15
Nodes (27): build_policy_scorecard(), build_scenario_policies(), main(), _parse_args(), policy_boundary(), Any, DataFrame, Namespace (+19 more)

### Community 1 - "SpatioTemporalGNN"
Cohesion: 0.12
Nodes (14): aggregate_neighbors(), CausalDecomposition, MLPBaseline, neighbor_exposure(), normalized_adjacency(), PersistenceBaseline, device, Tensor (+6 more)

### Community 2 - "run_forecast_experiment"
Cohesion: 0.20
Nodes (20): evaluate_model(), _is_trainable(), make_windows(), _predict(), device, ndarray, Tensor, row_models() (+12 more)

### Community 3 - "test_cgcen.py"
Cohesion: 0.23
Nodes (23): affordability(), compare_policies(), cumulative_rent_burden(), derive_low_income_share(), displacement_pressure_index(), _dpi_weights(), housing_transport_index(), housing_transport_overburden() (+15 more)

### Community 4 - "plot_generator.py"
Cohesion: 0.18
Nodes (22): build_model_suite(), treatment_vector(), apply_style(), degenerate_onset(), figure_ablation_topology(), figure_model_comparison(), figure_scenario_trajectories(), figure_spillover_heatmaps() (+14 more)

### Community 5 - "run_benchmark.py"
Cohesion: 0.25
Nodes (18): ablation_table(), confidence_interval(), main(), paired_test(), _parse_args(), Any, DataFrame, Namespace (+10 more)

### Community 6 - "latex_export.py"
Cohesion: 0.23
Nodes (14): build_algorithm_block(), build_benchmark_table(), build_scorecard_table(), escape(), format_p_value(), generate_all_latex(), main(), mean_std() (+6 more)

### Community 7 - "load_empirical_city"
Cohesion: 0.37
Nodes (12): build_panel(), build_tract_adjacency(), edge_index_from_adjacency(), _haversine_km(), load_empirical_city(), load_tract_table(), Any, ndarray (+4 more)

### Community 8 - "FeatureScaler"
Cohesion: 0.36
Nodes (4): prepare_splits(), FeatureScaler, ndarray, zscore()

### Community 11 - "synth.py"
Cohesion: 0.22
Nodes (13): _build_lattice_adjacency(), _coerce_config(), _edge_index_from_adjacency(), export_scenario_datasets(), _load_config(), load_scenario_dataset(), Any, ndarray (+5 more)

### Community 12 - "CivicTwin"
Cohesion: 0.17
Nodes (11): Causal decomposition, Causal decomposition, CivicTwin, Goals, Important caveat, Known limitations, Metrics, Quick start (+3 more)

### Community 13 - "Project Status (last updated 2026-08-31, commit `feed8f4`+)"
Cohesion: 0.18
Nodes (10): 1. Research Identity & Publication Goal, 2. Formal Mathematical Engine, 3. Execution Verification, 4. Implementation Status — COMPLETE, 5. Empirical Findings — RQ1 IS SUPPORTED, 6. Known Defects, 7. Recommended Next Step, CivicTwin: Causal Graph Counterfactual Exposure Network (CG-CEN) (+2 more)

### Community 14 - "build_graph"
Cohesion: 0.22
Nodes (8): build_graph(), _parse_args(), Any, Namespace, ndarray, Graph construction utilities for the CivicTwin synthetic panel., Encode the panel as a graph object. If `no_edges` is true, the graph retains…, test_build_graph_uses_isolated_nodes_when_no_edges()

## Knowledge Gaps
- **17 isolated node(s):** `1. Research Identity & Publication Goal`, `2. Formal Mathematical Engine`, `3. Execution Verification`, `4. Implementation Status — COMPLETE`, `5. Empirical Findings — RQ1 IS SUPPORTED` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SpatioTemporalGNN` connect `SpatioTemporalGNN` to `run_rq1.py`, `run_forecast_experiment`, `test_cgcen.py`, `plot_generator.py`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `run_forecast_experiment()` connect `run_forecast_experiment` to `run_rq1.py`, `SpatioTemporalGNN`, `plot_generator.py`, `run_benchmark.py`, `FeatureScaler`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `FeatureScaler` connect `FeatureScaler` to `run_rq1.py`, `run_forecast_experiment`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `SpatioTemporalGNN` (e.g. with `_predict()` and `run_forecast_experiment()`) actually correct?**
  _`SpatioTemporalGNN` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `1. Research Identity & Publication Goal`, `2. Formal Mathematical Engine`, `3. Execution Verification` to the rest of the system?**
  _17 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `SpatioTemporalGNN` be split into smaller, more focused modules?**
  _Cohesion score 0.12091038406827881 - nodes in this community are weakly interconnected._