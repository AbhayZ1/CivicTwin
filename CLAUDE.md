# CivicTwin: Causal Graph Counterfactual Exposure Network (CG-CEN)
## IEEE/ACM Journal-Level Development Guidelines

### 1. Research Identity & Publication Goal
- **Target Venues:** IEEE TKDE, ACM TSAS, Elsevier CEUS.
- **Novel Core:** Causal counterfactual scenario modeling with spatial interference (SUTVA violation handling) across urban policy interventions.
- **Scope Rule:** Non-predictive framing. Emphasize ex-ante comparative policy evaluation and spatial spillover identification over real-world point predictions.

### 2. Formal Mathematical Engine
- **Direct & Spillover Effect Splitting:**
  $$\hat{Y}_{i,t+1} = \Phi(X_{i,t}, P_{i,t}) + \sum_{j \in \mathcal{N}(i)} W_{ij} \cdot \Psi(X_{j,t}, P_{j,t})$$
  where \Phi estimates Individual Treatment Effect (ITE) and \Psi captures Spatial Interference Effect (STE).
- **Housing + Transportation Overburden Index:**
  $$H+T_{i,t} = \frac{\text{Rent}_{i,t} + \text{TransitCost}_{i,t}}{\text{MedianIncome}_{i,t}}$$
- **Displacement Pressure Index (DPI):**
  $$DPI_{i,t} = \alpha \left(\frac{\Delta \text{Rent}_{i,t}}{\text{Rent}_{i,t-1}}\right) + \beta \left(1 - \frac{\text{LowIncomeShare}_{i,t}}{\text{LowIncomeShare}_{i,t-1}}\right) + \gamma \left(\Delta \text{Accessibility}_{i,t}\right)$$

### 3. Execution Verification
- **Test Suite Command:** `python -m pytest -q`
- **Multi-Seed Benchmark Command:** `python -m civictwin.experiments.run_benchmark --seeds 10 --output_dir ./results`

---

## Project Status (last updated 2026-08-31, commit `ff79c7d`)

### 4. Implementation Status — COMPLETE
All five phases are implemented, benchmarked and pushed. `python -m pytest -q` → **17 passed**.

| Component | Location | State |
|---|---|---|
| Causal ST-GNN with Φ/Ψ split | `civictwin/model.py` | done |
| Baselines: MLP, Spatial Lag, Persistence | `civictwin/model.py` | done |
| Z-score scaling (train split only) | `civictwin/scaling.py` | done |
| H+T, DPI, MAE, true RMSE (vectorised) | `civictwin/evaluate.py` | done |
| Scenario dataset export (10 seeds × 4 scenarios) | `civictwin/synth.py` → `data/synthetic/` | done |
| NYC tract/PLUTO → PyG loader | `civictwin/data/empirical_loader.py` | done, **sample data only** |
| Multi-seed benchmark, CIs, paired t-tests, ablations | `civictwin/experiments/run_benchmark.py` → `results/` | done |
| Publication figures (300 DPI PDF+PNG) | `civictwin/visualization/plot_generator.py` → `paper_assets/figures/` | done |
| LaTeX tables + IEEE algorithm block | `civictwin/paper/latex_export.py` → `paper_assets/` | done |

Implementation notes that matter:
- `Φ` is anchored on the last observation (residual forecasting). Removing that anchor costs
  roughly an order of magnitude in MAE — do not "simplify" it away.
- `causal_effects()` contrasts each term against its `P = 0` counterfactual, so the ITE is
  exactly 0 on untreated nodes and the STE is exactly 0 on an edgeless graph.
- Python sources carry no `#` comments, per the code constraint.

### 5. Empirical Findings — RQ1 IS NOT SUPPORTED
10 seeds × 4 scenarios, pooled MAE on the real topology:

| Model | MAE | vs ST-GNN (paired t-test) |
|---|---|---|
| Spatial Lag Model | **6.92 ± 2.42** | ST-GNN significantly **worse**, p < 1e-5 |
| Causal ST-GNN | 14.82 ± 1.11 | — |
| Temporal MLP | 16.74 ± 3.47 | not significant, p = 0.10–0.19 |
| Naive Persistence | 60.04 ± 0.91 | ST-GNN wins, p < 1e-14 |

**No model derives measurable benefit from the graph.** Three independent checks:
1. ST-GNN with edges removed is *better* (14.16 vs 14.82); shuffled ≈ real (p = 0.70–0.93).
2. SLM real 6.92 ≈ no-edges 7.26 (p = 0.15) ≈ shuffled 6.47 (p = 0.14).
3. The SLM's learned spatial-lag coefficient ρ converges to ≈ 0 (−0.05, −0.07, 0.02).

Root cause: the accessibility shock is a **one-time permanent step** at `t ≥ 3` on fixed nodes.
By the evaluation window (`t ≥ 12`) each neighbour's contribution is a constant per-node offset,
already absorbed into the node's own level and recovered by the residual anchor. Spillover that
is permanent and early is not identifiable from spillover baked into the level.

Do not claim graph superiority in the paper on the current generator. Do not re-derive this;
it is settled by `results/ablation_topology.csv` and `results/benchmark_significance.csv`.

### 6. Known Defects — BLOCKING FOR SUBMISSION
1. **Generator degenerates before t = 100.** Income hits the hard floor `max(1000.0, ...)` in
   `synth.py` at `t = 52`; by `t = 100` all 16 nodes are pinned there while rent compounds to
   ~4e10. Anything past `t ≈ 52` measures the clamp, not the dynamics. Benchmarks use
   `n_steps = 18`, inside the stable regime; `scenario_trajectories.pdf` shades the dead zone.
2. **DPI cannot detect sustained policy effects.** As defined in §2 it is a growth-rate index,
   so a permanent level shift produces one transient spike and returns to zero. This is why
   every `pressure_delta` is ~1e-4. The implementation matches the formula; the formula is the
   limitation.
3. **No real empirical data.** `empirical_loader` ships a schema-compatible *sample*, never
   presented as real ACS/PLUTO. Supply a genuine extract via
   `load_empirical_city(table_path=...)`; `is_synthetic_sample` reports which source was used.

### 7. Recommended Next Step
Fix the generator, not the model: make neighbour shocks **recur during the evaluation window**
(time-varying rather than a single permanent step), and add an equilibrating supply response so
long horizons stay bounded. That is the only change that makes RQ1 testable. Re-running
`run_benchmark` after the change takes ~25 min on CPU.

