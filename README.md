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
- `civictwin/model.py`: graph-temporal and non-graph baseline models
- `civictwin/evaluate.py`: affordability and displacement scoring
- `civictwin/experiments/run_rq1.py`: RQ1 experiment script
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

## Synthetic assumptions

The synthetic generation is intentionally transparent and not meant to be a real policy model. It creates:

- accessibility shocks that affect specific neighborhoods
- land-value increases that spur rent growth with a lag
- spillover effects to nearby neighborhoods via graph adjacency
- income and housing changes that respond to price pressure

These dynamics are simple by design and are meant to validate the modeling pipeline, not to claim real-world validity.

## Important caveat

The displacement-pressure score is a constructed comparative indicator, not a validated prediction of actual displacement. It is a tool for comparing synthetic scenarios in a controlled setting.
