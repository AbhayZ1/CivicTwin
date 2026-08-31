from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from civictwin.policy import Policy, apply_policy

EPS = 1e-8

LAND_VALUE, RENT, POPULATION, INCOME, HOUSING_UNITS, ACCESSIBILITY = range(6)

DEFAULT_DPI_WEIGHTS: Dict[str, float] = {"alpha": 0.4, "beta": 0.3, "gamma": 0.3}

DEFAULT_HT_PARAMS: Dict[str, float] = {
    "transit_base_cost": 20.0,
    "transit_accessibility_elasticity": 1.0,
    "annualization": 1.0,
    "affordability_threshold": 0.45,
}


def _ht_params(params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    merged = dict(DEFAULT_HT_PARAMS)
    if params:
        merged.update({k: float(v) for k, v in params.items() if k in merged})
    return merged


def _dpi_weights(weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    merged = dict(DEFAULT_DPI_WEIGHTS)
    if weights:
        legacy = {"rent_growth": "alpha", "income_change": "beta", "housing_change": "gamma"}
        for key, value in weights.items():
            target = legacy.get(key, key)
            if target in merged:
                merged[target] = float(value)
    return merged


def affordability(panel: np.ndarray) -> np.ndarray:
    return panel[:, :, RENT] / np.maximum(panel[:, :, INCOME], 1.0)


def transit_cost(panel: np.ndarray, params: Optional[Dict[str, float]] = None) -> np.ndarray:
    cfg = _ht_params(params)
    accessibility = np.maximum(panel[:, :, ACCESSIBILITY], EPS)
    return cfg["transit_base_cost"] / np.power(
        accessibility, cfg["transit_accessibility_elasticity"]
    )


def housing_transport_index(
    panel: np.ndarray, params: Optional[Dict[str, float]] = None
) -> np.ndarray:
    cfg = _ht_params(params)
    rent = panel[:, :, RENT]
    income = np.maximum(panel[:, :, INCOME], EPS)
    return cfg["annualization"] * (rent + transit_cost(panel, cfg)) / income


def housing_transport_overburden(
    panel: np.ndarray,
    params: Optional[Dict[str, float]] = None,
    per_step: bool = False,
) -> np.ndarray:
    index = housing_transport_index(panel, params)
    if per_step:
        return index
    return index.mean(axis=1)


def overburden_rate(
    panel: np.ndarray, params: Optional[Dict[str, float]] = None
) -> np.ndarray:
    cfg = _ht_params(params)
    return (housing_transport_index(panel, cfg) > cfg["affordability_threshold"]).mean(axis=1)


def derive_low_income_share(panel: np.ndarray) -> np.ndarray:
    return np.clip(
        0.45
        + 0.18 * (panel[:, :, RENT] / np.maximum(panel[:, :, INCOME], 1.0))
        - 0.10 * (panel[:, :, HOUSING_UNITS] / np.maximum(panel[:, :, POPULATION], 1.0)),
        0.0,
        1.0,
    )


def displacement_pressure_index(
    panel: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
    low_income_share: Optional[np.ndarray] = None,
    per_step: bool = False,
) -> np.ndarray:
    merged = _dpi_weights(weights)

    if panel.shape[1] < 2:
        zeros = np.zeros((panel.shape[0], 0)) if per_step else np.zeros((panel.shape[0],))
        return zeros

    if low_income_share is None:
        low_income_share = derive_low_income_share(panel)
    low_income_share = np.asarray(low_income_share, dtype=np.float64)

    rent = panel[:, :, RENT]
    accessibility = panel[:, :, ACCESSIBILITY]

    rent_term = np.diff(rent, axis=1) / np.maximum(rent[:, :-1], EPS)
    share_term = 1.0 - low_income_share[:, 1:] / np.maximum(low_income_share[:, :-1], EPS)
    accessibility_term = np.diff(accessibility, axis=1)

    index = (
        merged["alpha"] * rent_term
        + merged["beta"] * share_term
        + merged["gamma"] * accessibility_term
    )
    if per_step:
        return index
    return index.mean(axis=1)


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(targets))))


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    error = np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, float]:
    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: predictions {pred.shape} vs targets {true.shape}")

    flat = (pred - true).reshape(-1, pred.shape[-1])
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
    merged_weights = _dpi_weights(weights)
    cfg = _ht_params(overburden_params)

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
    "transit_cost",
    "housing_transport_index",
    "housing_transport_overburden",
    "overburden_rate",
    "derive_low_income_share",
    "displacement_pressure_index",
    "compare_policies",
    "mae",
    "rmse",
    "regression_metrics",
    "DEFAULT_DPI_WEIGHTS",
    "DEFAULT_HT_PARAMS",
]
