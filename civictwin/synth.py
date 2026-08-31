"""Synthetic city generator for the CivicTwin research prototype.

This module creates a reproducible synthetic panel with a transparent generative
process: accessibility shocks raise land values, rents respond with a lag, and
neighbor spillovers diffuse across the graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "n_neighborhoods": 16,
    "n_steps": 12,
    "feature_names": [
        "land_value",
        "rent",
        "population",
        "median_income",
        "housing_units",
        "accessibility",
    ],
    "city": {
        "lattice_rows": 4,
        "lattice_cols": 4,
        "neighbor_decay": 0.35,
        "random_graph_p": 0.08,
    },
    "synth": {
        "accessibility_shock_strength": 2.0,
        "accessibility_shock_nodes": [0, 1, 4, 7],
        "accessibility_shock_step": 3,
        "land_value_to_rent": 0.45,
        "rent_lag": 1,
        "rent_spill_decay": 0.25,
        "neighbor_shock_transmission": 0.10,
        "pulse_rate": 0.22,
        "pulse_strength": 1.2,
        "pulse_start": 1,
        "pulse_transmission": 0.22,
        "logistic_damping": True,
        "land_carrying_multiple": 12.0,
        "income_carrying_multiple": 4.0,
        "rent_carrying_ratio": 0.012,
        "rent_target_ratio": 0.55,
        "rent_reversion": 0.05,
        "income_reversion": 0.02,
        "income_pressure": 0.1,
        "low_income_share_shift": 0.04,
        "population_growth": 0.08,
        "housing_growth": 0.03,
        "base_land_value": 100.0,
        "base_rent": 50.0,
        "base_population": 2000.0,
        "base_income": 60000.0,
        "base_housing_units": 900.0,
        "base_accessibility": 1.0,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base without dropping sibling keys.

    A shallow update would replace a whole nested block, so a partial override
    such as {"synth": {"pulse_rate": 0.5}} would silently discard every other
    synth key.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load config from YAML if available, else use the built-in defaults."""
    if path is not None:
        config_path = Path(path)
        if config_path.exists() and yaml is not None:
            with config_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if loaded:
                return _deep_merge(DEFAULT_CONFIG, loaded)

    default_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    if default_path.exists() and yaml is not None:
        with default_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if loaded:
            return _deep_merge(DEFAULT_CONFIG, loaded)
    return DEFAULT_CONFIG.copy()


def _build_lattice_adjacency(n_nodes: int, rows: int, cols: int) -> np.ndarray:
    """Create a simple 2D lattice adjacency matrix with undirected edges."""
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for node in range(n_nodes):
        r = node // cols
        c = node % cols
        neighbors = []
        if r > 0:
            neighbors.append(node - cols)
        if r + 1 < rows:
            neighbors.append(node + cols)
        if c > 0:
            neighbors.append(node - 1)
        if c + 1 < cols:
            neighbors.append(node + 1)
        for neighbor in neighbors:
            if 0 <= neighbor < n_nodes:
                adjacency[node, neighbor] = 1.0
                adjacency[neighbor, node] = 1.0
    return adjacency


def _edge_index_from_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Convert an adjacency matrix to a PyG-style edge_index array with shape [2, E]."""
    upper = np.triu(adjacency, k=1)
    rows, cols = np.nonzero(upper)
    edge_index = np.concatenate(
        [
            np.stack([rows, cols], axis=0),
            np.stack([cols, rows], axis=0),
        ],
        axis=1,
    )
    return edge_index.astype(np.int64)


def _coerce_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if config is None:
        return _load_config()
    merged = _load_config()
    if isinstance(config, dict):
        merged = _deep_merge(merged, config)
    return merged


def generate_synthetic_city(
    seed: int = 42,
    n_neighborhoods: int = 16,
    n_steps: int = 12,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a reproducible synthetic city panel and graph.

    The synthetic dynamics are intentionally transparent:
    - accessibility shocks create local uplift
    - land values react quickly
    - rents lag behind and are influenced by neighbor spillovers
    - income and housing adjust with price pressure
    """
    cfg = _coerce_config(config)
    rng = np.random.default_rng(seed)

    rows = int(cfg.get("city", {}).get("lattice_rows", 4))
    cols = int(cfg.get("city", {}).get("lattice_cols", 4))
    if n_neighborhoods is None:
        n_neighborhoods = int(cfg.get("n_neighborhoods", 16))
    if n_steps is None:
        n_steps = int(cfg.get("n_steps", 12))

    if n_neighborhoods <= 0:
        raise ValueError("n_neighborhoods must be positive")
    if n_steps <= 1:
        raise ValueError("n_steps must be at least 2")

    rows = max(1, int(np.ceil(np.sqrt(n_neighborhoods))))
    cols = max(1, int(np.ceil(n_neighborhoods / rows)))
    if rows * cols < n_neighborhoods:
        rows += 1

    adjacency = _build_lattice_adjacency(n_neighborhoods, rows, cols)
    edge_index = _edge_index_from_adjacency(adjacency)

    feature_names = list(cfg.get("feature_names", DEFAULT_CONFIG["feature_names"]))
    panel = np.zeros((n_neighborhoods, n_steps, len(feature_names)), dtype=np.float64)

    synth_cfg = cfg.get("synth", {})
    shock_nodes = set(synth_cfg.get("accessibility_shock_nodes", [0, 1, 4, 7]))
    shock_step = int(synth_cfg.get("accessibility_shock_step", 3))
    shock_strength = float(synth_cfg.get("accessibility_shock_strength", 2.0))
    neighbor_decay = float(cfg.get("city", {}).get("neighbor_decay", 0.35))
    shock_transmission = float(synth_cfg.get("neighbor_shock_transmission", 0.10))

    pulse_rate = float(synth_cfg.get("pulse_rate", 0.22))
    pulse_strength = float(synth_cfg.get("pulse_strength", 1.2))
    pulse_start = int(synth_cfg.get("pulse_start", 1))
    pulse_transmission = float(synth_cfg.get("pulse_transmission", 0.22))

    logistic_damping = bool(synth_cfg.get("logistic_damping", True))
    land_carrying_multiple = float(synth_cfg.get("land_carrying_multiple", 12.0))
    income_carrying_multiple = float(synth_cfg.get("income_carrying_multiple", 4.0))
    rent_carrying_ratio = float(synth_cfg.get("rent_carrying_ratio", 0.012))
    rent_target_ratio = float(synth_cfg.get("rent_target_ratio", 0.55))
    rent_reversion = float(synth_cfg.get("rent_reversion", 0.05))
    income_reversion = float(synth_cfg.get("income_reversion", 0.02))

    base_land_value = float(synth_cfg.get("base_land_value", 100.0))
    base_rent = float(synth_cfg.get("base_rent", 50.0))
    base_population = float(synth_cfg.get("base_population", 2000.0))
    base_income = float(synth_cfg.get("base_income", 60000.0))
    base_housing = float(synth_cfg.get("base_housing_units", 900.0))
    base_accessibility = float(synth_cfg.get("base_accessibility", 1.0))

    for node in range(n_neighborhoods):
        panel[node, 0, 0] = base_land_value * (1.0 + 0.08 * rng.normal())
        panel[node, 0, 1] = base_rent * (1.0 + 0.05 * rng.normal())
        panel[node, 0, 2] = base_population * (1.0 + 0.05 * rng.normal())
        panel[node, 0, 3] = base_income * (1.0 + 0.06 * rng.normal())
        panel[node, 0, 4] = base_housing * (1.0 + 0.04 * rng.normal())
        panel[node, 0, 5] = base_accessibility + 0.05 * rng.normal()

    pulse = np.zeros((n_neighborhoods, n_steps), dtype=np.float64)
    if pulse_rate > 0.0 and pulse_strength > 0.0:
        active = rng.random((n_neighborhoods, n_steps)) < pulse_rate
        magnitude = rng.uniform(
            pulse_strength * 0.5, pulse_strength * 1.5, size=(n_neighborhoods, n_steps)
        )
        pulse = active * magnitude
        pulse[:, :pulse_start] = 0.0

    land_capacity = base_land_value * land_carrying_multiple
    income_capacity = base_income * income_carrying_multiple

    for step in range(1, n_steps):
        local_neighbor_land = adjacency @ panel[:, step - 1, 0]
        local_neighbor_rent = adjacency @ panel[:, step - 1, 1]
        degrees = adjacency.sum(axis=1)
        neighbor_land_term = local_neighbor_land / np.maximum(degrees, 1)
        neighbor_rent_term = local_neighbor_rent / np.maximum(degrees, 1)

        neighbor_shock_term = (
            adjacency @ (panel[:, step - 1, 5] - base_accessibility)
        ) / np.maximum(degrees, 1)

        neighbor_pulse_term = (adjacency @ pulse[:, step - 1]) / np.maximum(degrees, 1)

        for node in range(n_neighborhoods):
            shock_indicator = 1.0 if node in shock_nodes and step >= shock_step else 0.0
            accessibility = (
                base_accessibility
                + shock_strength * shock_indicator
                + pulse[node, step]
                + 0.03 * np.sin(step + node) * (0.5 + 0.5 * rng.random())
            )
            panel[node, step, 5] = accessibility

            prev_land = panel[node, step - 1, 0]
            prev_rent = panel[node, step - 1, 1]
            prev_population = panel[node, step - 1, 2]
            prev_income = panel[node, step - 1, 3]
            prev_housing = panel[node, step - 1, 4]

            rent_capacity = max(rent_carrying_ratio * prev_income, 1.0)
            land_headroom = max(0.0, 1.0 - prev_land / land_capacity) if logistic_damping else 1.0
            rent_headroom = max(0.0, 1.0 - prev_rent / rent_capacity) if logistic_damping else 1.0
            income_headroom = (
                max(0.0, 1.0 - prev_income / income_capacity) if logistic_damping else 1.0
            )

            spillover = neighbor_decay * neighbor_land_term[node]
            land_growth = 0.02 + 0.05 * accessibility + 0.12 * spillover / max(prev_land, 1.0)
            panel[node, step, 0] = max(5.0, prev_land * (1.0 + land_growth * land_headroom))

            rent_growth = (
                0.01
                + 0.12 * (panel[node, step, 0] - prev_land) / max(prev_land, 1.0)
                + 0.25 * neighbor_decay * neighbor_rent_term[node] / max(prev_rent, 1.0)
                + shock_transmission * neighbor_shock_term[node]
                + pulse_transmission * neighbor_pulse_term[node]
            )
            panel[node, step, 1] = max(
                1.0,
                prev_rent
                * (1.0 + rent_growth * rent_headroom)
                + rent_reversion * (rent_capacity * rent_target_ratio - prev_rent),
            )

            burden = panel[node, step, 1] / max(prev_income, 1.0)
            population_growth = 0.01 + 0.02 * accessibility - 0.04 * burden
            panel[node, step, 2] = max(100.0, prev_population * (1.0 + population_growth))

            income_growth = 0.005 + 0.02 * accessibility - 0.03 * burden
            panel[node, step, 3] = max(
                1000.0,
                prev_income * (1.0 + income_growth * income_headroom)
                + income_reversion * (base_income - prev_income),
            )

            housing_growth = 0.01 + 0.02 * accessibility - 0.01 * (panel[node, step, 1] / max(prev_rent, 1.0))
            panel[node, step, 4] = max(50.0, prev_housing * (1.0 + housing_growth))

    low_income_share = np.clip(
        0.45
        + 0.18 * (panel[..., 1] / np.maximum(panel[..., 3], 1.0))
        - 0.10 * (panel[..., 4] / np.maximum(panel[..., 2], 1.0)),
        0.0,
        1.0,
    )

    return {
        "panel": panel,
        "adjacency": adjacency,
        "edge_index": edge_index,
        "feature_names": feature_names,
        "low_income_share": low_income_share,
        "config": cfg,
        "n_neighborhoods": n_neighborhoods,
        "n_steps": n_steps,
    }


def scenario_panels(
    seed: int = 42,
    n_neighborhoods: int = 16,
    n_steps: int = 18,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from civictwin.policy import (
        apply_policy,
        baseline_policy,
        inclusionary_housing_policy,
        land_value_capture_policy,
        market_led_policy,
    )

    city = generate_synthetic_city(
        seed=seed, n_neighborhoods=n_neighborhoods, n_steps=n_steps, config=config
    )
    cfg = city["config"]
    synth_cfg = cfg.get("synth", {}) or {}
    policy_cfg = cfg.get("policy", {}) or {}
    shock_nodes = synth_cfg.get("accessibility_shock_nodes", [0, 1, 4, 7])
    boundary = sorted({int(n) for n in shock_nodes if 0 <= int(n) < n_neighborhoods}) or [0]

    def intensity(name: str, fallback: float) -> float:
        return float((policy_cfg.get(name, {}) or {}).get("intensity", fallback))

    policies = [
        baseline_policy(),
        market_led_policy(boundary, timing=2, intensity=intensity("market_led", 0.8)),
        inclusionary_housing_policy(
            boundary, timing=1, intensity=intensity("inclusionary_housing", 0.6)
        ),
        land_value_capture_policy(
            boundary, timing=2, intensity=intensity("land_value_capture", 0.7)
        ),
    ]

    panels = {p.name: apply_policy(city["panel"], p) for p in policies}
    city["scenarios"] = panels
    city["boundary"] = boundary
    return city


def export_scenario_datasets(
    seeds: Iterable[int] = tuple(range(1, 11)),
    n_neighborhoods: int = 16,
    n_steps: int = 18,
    output_dir: str | Path = "./data/synthetic",
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    manifest = []
    for seed in seeds:
        city = scenario_panels(
            seed=int(seed),
            n_neighborhoods=n_neighborhoods,
            n_steps=n_steps,
            config=config,
        )
        for scenario, panel in city["scenarios"].items():
            filename = f"seed{int(seed):03d}_{scenario}.npz"
            np.savez_compressed(
                destination / filename,
                panel=panel,
                adjacency=city["adjacency"],
                edge_index=city["edge_index"],
                low_income_share=city["low_income_share"],
                feature_names=np.array(city["feature_names"], dtype=object),
                boundary=np.array(city["boundary"], dtype=np.int64),
                seed=np.array([int(seed)]),
            )
            manifest.append(
                {
                    "seed": int(seed),
                    "scenario": scenario,
                    "file": filename,
                    "n_neighborhoods": int(panel.shape[0]),
                    "n_steps": int(panel.shape[1]),
                    "n_features": int(panel.shape[2]),
                    "n_edges": int(city["edge_index"].shape[1]),
                    "treated_nodes": int(len(city["boundary"])),
                }
            )

    import csv

    manifest_path = destination / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    return manifest_path


def load_scenario_dataset(path: str | Path) -> Dict[str, Any]:
    data = np.load(Path(path), allow_pickle=True)
    return {
        "panel": data["panel"],
        "adjacency": data["adjacency"],
        "edge_index": data["edge_index"],
        "low_income_share": data["low_income_share"],
        "feature_names": list(data["feature_names"]),
        "boundary": list(data["boundary"]),
        "seed": int(data["seed"][0]),
    }


__all__ = [
    "generate_synthetic_city",
    "scenario_panels",
    "export_scenario_datasets",
    "load_scenario_dataset",
    "_load_config",
]
