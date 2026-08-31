# Graph Report - CivicTwin  (2026-08-31)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 227 nodes · 610 edges · 11 communities (9 shown, 2 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `feed8f4c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- synth.py
- SpatioTemporalGNN
- run_rq1.py
- test_cgcen.py
- plot_generator.py
- run_benchmark.py
- latex_export.py
- load_empirical_city
- FeatureScaler
- experiments/__init__.py
- civictwin/__init__.py

## God Nodes (most connected - your core abstractions)
1. `run_forecast_experiment()` - 23 edges
2. `SpatioTemporalGNN` - 22 edges
3. `generate_synthetic_city()` - 18 edges
4. `scenario_panels()` - 15 edges
5. `Policy` - 14 edges
6. `compare_policies()` - 14 edges
7. `displacement_pressure_index()` - 14 edges
8. `main()` - 13 edges
9. `apply_policy()` - 12 edges
10. `train_model()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `test_stgnn_decomposes_into_direct_and_spillover_terms()` --calls--> `SpatioTemporalGNN`  [EXTRACTED]
  tests/test_cgcen.py → civictwin/model.py
- `test_stgnn_training_reduces_error_versus_untrained_model()` --calls--> `SpatioTemporalGNN`  [EXTRACTED]
  tests/test_rq1.py → civictwin/model.py
- `test_build_graph_uses_isolated_nodes_when_no_edges()` --calls--> `build_graph()`  [EXTRACTED]
  tests/test_graph_policy_eval.py → civictwin/graph.py
- `test_dpi_keeps_permanent_policy_shift_visible_at_long_horizon()` --calls--> `scenario_panels()`  [EXTRACTED]
  tests/test_cgcen.py → civictwin/synth.py
- `test_scenario_panels_cover_all_four_policies()` --calls--> `scenario_panels()`  [EXTRACTED]
  tests/test_cgcen.py → civictwin/synth.py

## Import Cycles
- None detected.

## Communities (11 total, 2 thin omitted)

### Community 0 - "synth.py"
Cohesion: 0.09
Nodes (38): build_scenario_policies(), build_graph(), _parse_args(), Any, Namespace, ndarray, Graph construction utilities for the CivicTwin synthetic panel., Encode the panel as a graph object. If `no_edges` is true, the graph retains… (+30 more)

### Community 1 - "SpatioTemporalGNN"
Cohesion: 0.12
Nodes (14): aggregate_neighbors(), CausalDecomposition, MLPBaseline, neighbor_exposure(), normalized_adjacency(), PersistenceBaseline, device, Tensor (+6 more)

### Community 2 - "run_rq1.py"
Cohesion: 0.16
Nodes (31): build_policy_scorecard(), evaluate_model(), _is_trainable(), main(), make_windows(), _parse_args(), policy_boundary(), _predict() (+23 more)

### Community 3 - "test_cgcen.py"
Cohesion: 0.25
Nodes (22): affordability(), compare_policies(), cumulative_rent_burden(), derive_low_income_share(), displacement_pressure_index(), _dpi_weights(), housing_transport_index(), housing_transport_overburden() (+14 more)

### Community 4 - "plot_generator.py"
Cohesion: 0.18
Nodes (22): build_model_suite(), treatment_vector(), apply_style(), degenerate_onset(), figure_ablation_topology(), figure_model_comparison(), figure_scenario_trajectories(), figure_spillover_heatmaps() (+14 more)

### Community 5 - "run_benchmark.py"
Cohesion: 0.23
Nodes (19): ablation_table(), confidence_interval(), main(), paired_test(), _parse_args(), Any, DataFrame, Namespace (+11 more)

### Community 6 - "latex_export.py"
Cohesion: 0.23
Nodes (14): build_algorithm_block(), build_benchmark_table(), build_scorecard_table(), escape(), format_p_value(), generate_all_latex(), main(), mean_std() (+6 more)

### Community 7 - "load_empirical_city"
Cohesion: 0.37
Nodes (12): build_panel(), build_tract_adjacency(), edge_index_from_adjacency(), _haversine_km(), load_empirical_city(), load_tract_table(), Any, ndarray (+4 more)

### Community 8 - "FeatureScaler"
Cohesion: 0.40
Nodes (3): FeatureScaler, ndarray, zscore()

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SpatioTemporalGNN` connect `SpatioTemporalGNN` to `run_rq1.py`, `test_cgcen.py`, `plot_generator.py`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **Why does `run_forecast_experiment()` connect `run_rq1.py` to `synth.py`, `SpatioTemporalGNN`, `plot_generator.py`, `run_benchmark.py`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `FeatureScaler` connect `FeatureScaler` to `run_rq1.py`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `SpatioTemporalGNN` (e.g. with `_predict()` and `run_forecast_experiment()`) actually correct?**
  _`SpatioTemporalGNN` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `synth.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08637873754152824 - nodes in this community are weakly interconnected._
- **Should `SpatioTemporalGNN` be split into smaller, more focused modules?**
  _Cohesion score 0.12091038406827881 - nodes in this community are weakly interconnected._