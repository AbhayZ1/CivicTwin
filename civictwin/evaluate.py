"""Metric engine for policy comparison and forecast scoring.

Two constructed indices are defined here. Both are fully vectorised over the
panel ``(N, T, F)`` -- no Python-level loops over nodes or time steps -- so they
can be evaluated for every scenario x seed combination cheaply.

H+T Overburden Index
--------------------
Following the combined Housing + Transportation affordability convention, a
household is *overburdened* when combined housing and transport costs exceed a
share ``theta`` of income (the CNT benchmark is ``theta = 0.45``).

    H_it  =  months * rent_to_monthly_cost * rent_it            (annual housing)
    T_it  =  transport_base_cost / max(accessibility_it, eps)^kappa
    s_it  =  ( H_it + T_it ) / max(income_it, eps)              (H+T share)
    O_it  =  max( s_it - theta , 0 )                            (overburden)

    HT_i  =  mean_t O_it

Accessibility enters through ``T`` with a negative elasticity: better-connected
neighbourhoods spend less on transport, which is precisely the mechanism a
housing-only affordability ratio misses.

Displacement Pressure Index (DPI)
---------------------------------
A comparative index blending rent acceleration against income and housing-stock
erosion. Growth rates are taken against the *previous* period:

    g^x_it  =  ( x_it - x_i,t-1 ) / max( x_i,t-1 , eps )

    DPI_i  =  w_r * mean_t[ max(  g^rent_it  , 0) ]
           +  w_y * mean_t[ max( -g^income_it, 0) ]
           +  w_h * mean_t[ max( -g^house_it , 0) ]

Only adverse movements contribute: rent rises, income falls, stock contracts.

Caveat: both indices are constructed comparative indicators over a synthetic
panel. Neither is a calibrated prediction of real displacement.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from civictwin.policy import Policy, apply_policy

EPS = 1e-8

# Panel feature axis positions (see synth.DEFAULT_CONFIG["feature_names"]).
LAND_VALUE, RENT, POPULATION, INCOME, HOUSING_UNITS, ACCESSIBILITY = range(6)

DEFAULT_DPI_WEIGHTS: Dict[str, float] = {
    "rent_growth": 0.4,
    "income_change": 0.3,
    "housing_change": 0.3,
}

#: Cost parameters for the H+T overburden index. ``rent_to_monthly_cost``
#: converts the synthetic rent *index* into a monthly currency amount; adjust it
#: (or ``threshold``) here rather than editing the formula.
DEFAULT_OVERBURDEN_PARAMS: Dict[str, float] = {
    "months_per_year": 12.0,
    "rent_to_monthly_cost": 30.0,
    "transport_base_cost": 6000.0,
    "transport_accessibility_elasticity": 1.0,
    "threshold": 0.45,
}


def _overburden_params(params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    merged = dict(DEFAULT_OVERBURDEN_PARAMS)
    if params:
        merged.update({k: float(v) for k, v in params.items() if k in merged})
    return merged


def affordability(panel: np.ndarray) -> np.ndarray:
    """Rent-to-income ratio for each node and time step, shape ``(N, T)``."""
    income = panel[:, :, INCOME]
    rent = panel[:, :, RENT]
    return rent / np.maximum(income, 1.0)


def housing_cost(panel: np.ndarray, params: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Annualised housing cost ``H_it``, shape ``(N, T)``."""
    cfg = _overburden_params(params)
    return panel[:, :, RENT] * cfg["rent_to_monthly_cost"] * cfg["months_per_year"]


def transport_cost(panel: np.ndarray, params: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Annualised transport cost ``T_it``, decreasing in accessibility."""
    cfg = _overburden_params(params)
    accessibility = np.maximum(panel[:, :, ACCESSIBILITY], EPS)
    return cfg["transport_base_cost"] / np.power(
        accessibility, cfg["transport_accessibility_elasticity"]
    )


def housing_transport_share(
    panel: np.ndarray, params: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """Combined H+T cost share of income ``s_it``, shape ``(N, T)``."""
    income = np.maximum(panel[:, :, INCOME], EPS)
    return (housing_cost(panel, params) + transport_cost(panel, params)) / income


def housing_transport_overburden(
    panel: np.ndarray,
    params: Optional[Dict[str, float]] = None,
    per_step: bool = False,
) -> np.ndarray:
    """H+T Overburden Index.

    Returns the node-level index ``(N,)`` -- the mean excess of the H+T share
    over the affordability threshold -- or the raw ``(N, T)`` surface when
    ``per_step`` is set.
    """
    cfg = _overburden_params(params)
    overburden = np.clip(housing_transport_share(panel, cfg) - cfg["threshold"], 0.0, None)
    if per_step:
        return overburden
    return overburden.mean(axis=1)


def overburden_rate(
    panel: np.ndarray, params: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """Share of time steps in which each node is H+T overburdened, ``(N,)``."""
    cfg = _overburden_params(params)
    return (housing_transport_share(panel, cfg) > cfg["threshold"]).mean(axis=1)


def _growth(series: np.ndarray) -> np.ndarray:
    """Period-over-period growth against the previous value, shape ``(N, T-1)``."""
    previous = series[:, :-1]
    return (series[:, 1:] - previous) / np.maximum(np.abs(previous), EPS)


def displacement_pressure_index(
    panel: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Displacement Pressure Index per node, shape ``(N,)``.

    Fully vectorised: growth rates, one-sided clipping and the weighted blend are
    all array operations over the ``(N, T)`` panel slices.
    """
    merged = dict(DEFAULT_DPI_WEIGHTS)
    if weights:
        merged.update(weights)

    if panel.shape[1] < 2:
        return np.zeros((panel.shape[0],))

    # Stack the three adverse-movement components into one (3, N, T-1) tensor so
    # the blend is a single tensordot rather than three separate reductions.
    components = np.stack(
        [
            np.clip(_growth(panel[:, :, RENT]), 0.0, None),
            np.clip(-_growth(panel[:, :, INCOME]), 0.0, None),
            np.clip(-_growth(panel[:, :, HOUSING_UNITS]), 0.0, None),
        ],
        axis=0,
    ).mean(axis=2)

    weight_vector = np.array(
        [
            merged.get("rent_growth", 0.4),
            merged.get("income_change", 0.3),
            merged.get("housing_change", 0.3),
        ],
        dtype=np.float64,
    )
    return weight_vector @ components


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean absolute error over every element."""
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(targets))))


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root mean squared error: ``sqrt(mean(err^2))`` over every element.

    Computed from the squared errors themselves -- never from an intermediate
    mean absolute error, which would understate the penalty on large residuals.
    """
    error = np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, float]:
    """Per-feature and overall MAE / RMSE.

    ``predictions`` and ``targets`` are ``(S, N, F)`` stacks of forecast windows.
    Errors are pooled across samples and nodes before reduction, so the reported
    RMSE is a true RMSE over the evaluation set.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: predictions {pred.shape} vs targets {true.shape}")

    error = pred - true
    flat = error.reshape(-1, error.shape[-1])

    metrics: Dict[str, float] = {}
    for idx, name in enumerate(feature_names):
        column = flat[:, idx]
        metrics[f"{name}_mae"] = float(np.mean(np.abs(column)))
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean(np.square(column))))

    metrics["overall_mae"] = float(np.mean(np.abs(flat)))
    metrics["overall_rmse"] = float(np.sqrt(np.mean(np.square(flat))))
    return metrics


def compare_policies(
    baseline_panel: np.ndarray,
    policies: Iterable[Policy],
    weights: Optional[Dict[str, float]] = None,
    overburden_params: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Score each policy against the untreated baseline panel.

    Ranking is ascending on ``pressure_delta``: the most negative change in
    displacement pressure ranks first.
    """
    merged_weights = dict(DEFAULT_DPI_WEIGHTS)
    if weights:
        merged_weights.update(weights)
    cfg = _overburden_params(overburden_params)

    # Baseline quantities are invariant across policies -- compute them once.
    base_affordability = float(affordability(baseline_panel).mean())
    base_pressure = float(displacement_pressure_index(baseline_panel, merged_weights).mean())
    base_overburden = float(housing_transport_overburden(baseline_panel, cfg).mean())

    rows = []
    for policy in policies:
        modified = apply_policy(baseline_panel, policy)
        mod_affordability = float(affordability(modified).mean())
        mod_pressure = float(displacement_pressure_index(modified, merged_weights).mean())
        mod_overburden = float(housing_transport_overburden(modified, cfg).mean())

        rows.append(
            {
                "policy": policy.name,
                "affordability_baseline": base_affordability,
                "affordability_policy": mod_affordability,
                "pressure_baseline": base_pressure,
                "pressure_policy": mod_pressure,
                "overburden_baseline": base_overburden,
                "overburden_policy": mod_overburden,
                "affordability_delta": mod_affordability - base_affordability,
                "pressure_delta": mod_pressure - base_pressure,
                "overburden_delta": mod_overburden - base_overburden,
            }
        )

    frame = pd.DataFrame(rows)
    frame["ranking"] = frame["pressure_delta"].rank(method="dense", ascending=True)
    return frame


__all__ = [
    "affordability",
    "housing_cost",
    "transport_cost",
    "housing_transport_share",
    "housing_transport_overburden",
    "overburden_rate",
    "displacement_pressure_index",
    "compare_policies",
    "mae",
    "rmse",
    "regression_metrics",
    "DEFAULT_DPI_WEIGHTS",
    "DEFAULT_OVERBURDEN_PARAMS",
]
