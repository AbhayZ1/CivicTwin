"""Evaluation utilities for policy comparison and multi-objective scoring.

This module is intentionally transparent: the displacement-pressure index is a
constructed comparative score, not a validated prediction of displacement.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from civictwin.policy import Policy, apply_policy


def affordability(panel: np.ndarray) -> np.ndarray:
    """Compute rent-to-income ratio for each node and time step."""
    income = panel[:, :, 3]
    rent = panel[:, :, 1]
    return rent / np.maximum(income, 1.0)


def displacement_pressure_index(
    panel: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Construct a transparent displacement-pressure index.

    The score blends rent growth, income change, and housing-stock change. It is
    a comparative synthetic indicator only and is not a calibrated measure of
    real displacement risk.
    """
    if weights is None:
        weights = {"rent_growth": 0.4, "income_change": 0.3, "housing_change": 0.3}

    if panel.shape[1] < 2:
        return np.zeros((panel.shape[0],))

    rent_growth = np.diff(panel[:, :, 1], axis=1) / np.maximum(panel[:, 1:, 1], 1.0)
    income_change = np.diff(panel[:, :, 3], axis=1) / np.maximum(panel[:, 1:, 3], 1.0)
    housing_change = np.diff(panel[:, :, 4], axis=1) / np.maximum(panel[:, 1:, 4], 1.0)

    rent_component = np.mean(np.clip(rent_growth, 0.0, None), axis=1)
    income_component = np.mean(np.clip(-income_change, 0.0, None), axis=1)
    housing_component = np.mean(np.clip(-housing_change, 0.0, None), axis=1)

    index = (
        weights.get("rent_growth", 0.4) * rent_component
        + weights.get("income_change", 0.3) * income_component
        + weights.get("housing_change", 0.3) * housing_component
    )
    return index


def compare_policies(
    baseline_panel: np.ndarray,
    policies: Iterable[Policy],
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compare policies against the baseline on the multi-objective scorecard."""
    rows = []
    if weights is None:
        weights = {"rent_growth": 0.4, "income_change": 0.3, "housing_change": 0.3}

    for policy in policies:
        modified = apply_policy(baseline_panel, policy)
        base_affordability = affordability(baseline_panel).mean()
        mod_affordability = affordability(modified).mean()
        base_pressure = displacement_pressure_index(baseline_panel, weights).mean()
        mod_pressure = displacement_pressure_index(modified, weights).mean()

        rows.append(
            {
                "policy": policy.name,
                "affordability_baseline": float(base_affordability),
                "affordability_policy": float(mod_affordability),
                "pressure_baseline": float(base_pressure),
                "pressure_policy": float(mod_pressure),
                "affordability_delta": float(mod_affordability - base_affordability),
                "pressure_delta": float(mod_pressure - base_pressure),
            }
        )

    frame = pd.DataFrame(rows)
    frame["ranking"] = frame["pressure_delta"].rank(method="dense", ascending=True)
    return frame


__all__ = ["affordability", "displacement_pressure_index", "compare_policies"]
