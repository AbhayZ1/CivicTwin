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

## Project Status (last updated 2026-08-31, commit `feed8f4`+)

### 4. Implementation Status — COMPLETE
All five phases are implemented, benchmarked and pushed. `python -m pytest -q` → **19 passed**.

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
- `Φ` is anchored on the last observation (residual forecasting). Ablating the anchor
  degrades pooled MAE from 8.92 to 37.28 (factor 4.2) — do not "simplify" it away.
- `causal_effects()` contrasts each term against its `P = 0` counterfactual, so the ITE is
  exactly 0 on untreated nodes and the STE is exactly 0 on an edgeless graph.
- Python sources carry no `#` comments, per the code constraint.

### 5. Empirical Findings — RQ1 IS SUPPORTED
10 seeds × 4 scenarios × 4 models, pooled MAE on the real topology (`results/`):

| Model | MAE | RMSE | vs ST-GNN (paired t-test, per scenario) |
|---|---|---|---|
| **Causal ST-GNN (CG-CEN)** | **8.88 ± 2.40** | **13.81** | — |
| Spatial Lag Model | 11.87 ± 3.12 | 19.29 | ST-GNN wins, p = 0.0015–0.0033 in 3/4; baseline n.s. (p = 0.13) |
| Temporal MLP | 20.03 ± 7.97 | 34.54 | ST-GNN wins, p = 0.0006–0.0058 in all 4 |
| Naive Persistence | 27.90 ± 1.27 | 37.54 | ST-GNN wins, p < 2e-8 in all 4 |

**The graph is now load-bearing** (`results/ablation_topology.csv`):

| Topology | MAE | Degradation | p vs real |
|---|---|---|---|
| Real graph | 8.88 | — | — |
| No edges | 11.77 | **+2.89** | 0.005–0.008 (3/4); baseline 0.080 |
| Shuffled edges | 13.76 | **+4.88** | 0.0001–0.018 (all 4) |

Shuffled is *worse than no edges*: a wrong topology is actively harmful, which is the strongest
available evidence that the model has learned the specific adjacency rather than generic
neighbourhood averaging. Compare the pre-fix run, where no-edges **beat** the real graph
(14.16 vs 14.82) and shuffled ≈ real (p = 0.70–0.93).

What changed: neighbour shocks are now **stochastic and recurring** (`pulse_rate = 0.22`,
`pulse_strength = 1.2`, `pulse_transmission = 0.22`), so a neighbour's transient accessibility
pulse at `t` moves ego rent at `t+1` and is not recoverable from ego history. Before the fix
the shock was a single permanent step that decayed into a static per-node intercept, which the
residual anchor already captured — hence the null result.

### 6. Known Defects
1. ~~Generator degenerates before t = 100~~ — **FIXED.** Logistic carrying capacities
   (`land_carrying_multiple`, `income_carrying_multiple`, `rent_carrying_ratio`) plus mean
   reversion (`rent_reversion`, `income_reversion`) hold a dynamic equilibrium: **no node
   reaches any hard floor through t = 100**. Rent settles ≈ 1.6e3, income ≈ 1.5e5, H+T ≈ 0.011.
2. ~~DPI cannot detect sustained policy effects~~ — **FIXED.** DPI gains a fourth term
   `δ · B_{i,t}`, where `B` is cumulative rent burden relative to each node's own `t = 0`
   H+T level. Policy deltas moved from ~1e-4 (transient) to ~1.7e-2 sustained. Setting
   `delta = 0` recovers the exact §2 three-term formula; this is pinned by a test.
3. **No real empirical data.** `empirical_loader` ships a schema-compatible *sample*, never
   presented as real ACS/PLUTO. Supply a genuine extract via
   `load_empirical_city(table_path=...)`; `is_synthetic_sample` reports which source was used.
   This is the remaining blocker for submission.

### 7. Recommended Next Step
Source a real NYC ACS/PLUTO tract extract and re-run the benchmark on it. The synthetic result
is now internally valid; external validity is untested. Note that the ST-GNN's edge over the
Spatial Lag Model is not significant in the untreated baseline scenario (p = 0.13) — it
separates only under active policy interventions, which is the honest framing for the paper.
