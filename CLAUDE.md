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