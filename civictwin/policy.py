"""Policy interventions for the synthetic urban prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class Policy:
    """A simple node-level policy intervention.

    boundary: which nodes are affected
    timing: at which time step the intervention starts
    intensity: scalar effect magnitude
    """

    name: str
    boundary: Sequence[int] = field(default_factory=lambda: [0])
    timing: int = 0
    intensity: float = 1.0

    def __post_init__(self) -> None:
        self.boundary = list(self.boundary)


def baseline_policy() -> Policy:
    return Policy(name="baseline", boundary=[], timing=0, intensity=0.0)


def market_led_policy(boundary: Iterable[int], timing: int = 2, intensity: float = 1.0) -> Policy:
    """Market-led policy: boost values and rents in selected neighborhoods."""
    return Policy(name="market_led", boundary=list(boundary), timing=timing, intensity=float(intensity))


def inclusionary_housing_policy(boundary: Iterable[int], timing: int = 1, intensity: float = 1.0) -> Policy:
    """Inclusionary housing: increase affordable stock and damp rent growth."""
    return Policy(name="inclusionary_housing", boundary=list(boundary), timing=timing, intensity=float(intensity))


def land_value_capture_policy(boundary: Iterable[int], timing: int = 2, intensity: float = 1.0) -> Policy:
    """Land value capture: taxes or mechanisms that partially offset price growth."""
    return Policy(name="land_value_capture", boundary=list(boundary), timing=timing, intensity=float(intensity))


def apply_policy(panel: np.ndarray, policy: Policy) -> np.ndarray:
    """Apply a policy to a panel, returning a modified copy.

    The interventions are intentionally simple and additive to keep the scaffold
    transparent and easy to test.
    """
    updated = panel.copy()
    if policy.name == "baseline":
        return updated

    if policy.intensity == 0:
        return updated

    start_step = max(0, int(policy.timing))
    for node in policy.boundary:
        for step in range(start_step, updated.shape[1]):
            if policy.name == "market_led":
                updated[node, step, 0] *= 1.0 + 0.04 * policy.intensity
                updated[node, step, 1] *= 1.0 + 0.06 * policy.intensity
            elif policy.name == "inclusionary_housing":
                updated[node, step, 1] *= 1.0 - 0.06 * policy.intensity
                updated[node, step, 4] *= 1.0 + 0.08 * policy.intensity
            elif policy.name == "land_value_capture":
                updated[node, step, 0] *= 1.0 + 0.02 * policy.intensity
                updated[node, step, 1] *= 1.0 - 0.05 * policy.intensity
            else:
                raise ValueError(f"Unsupported policy: {policy.name}")

    return updated


__all__ = [
    "Policy",
    "apply_policy",
    "baseline_policy",
    "market_led_policy",
    "inclusionary_housing_policy",
    "land_value_capture_policy",
]
