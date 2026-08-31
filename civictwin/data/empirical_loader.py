from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None

NYC_TRACT_SCHEMA: Dict[str, str] = {
    "geoid": "11-digit census tract FIPS code",
    "borough": "NYC borough name",
    "latitude": "tract centroid latitude (EPSG:4326)",
    "longitude": "tract centroid longitude (EPSG:4326)",
    "median_gross_rent": "ACS B25064 median gross rent, monthly USD",
    "median_household_income": "ACS B19013 median household income, annual USD",
    "population": "ACS B01003 total population",
    "housing_units": "ACS B25001 total housing units",
    "assessed_land_value": "PLUTO AssessLand aggregated to tract, USD",
    "lot_area": "PLUTO LotArea aggregated to tract, square feet",
    "residential_area": "PLUTO ResArea aggregated to tract, square feet",
    "transit_access_score": "derived subway/bus accessibility score, unitless",
}

PANEL_FEATURE_ORDER = [
    "assessed_land_value",
    "median_gross_rent",
    "population",
    "median_household_income",
    "housing_units",
    "transit_access_score",
]

_SAMPLE_BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]


def write_sample_tract_table(
    path: str | Path,
    n_tracts: int = 48,
    seed: int = 42,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows: List[Dict[str, Any]] = []
    for index in range(n_tracts):
        borough = _SAMPLE_BOROUGHS[index % len(_SAMPLE_BOROUGHS)]
        latitude = 40.62 + 0.16 * rng.random()
        longitude = -74.02 + 0.34 * rng.random()
        transit = float(np.clip(rng.normal(1.0, 0.35), 0.15, 3.0))
        income = float(np.clip(rng.normal(68000, 24000), 18000, 250000))
        rent = float(np.clip(rng.normal(1750, 620), 600, 6000))
        population = float(np.clip(rng.normal(4200, 1400), 400, 12000))
        units = float(np.clip(population / rng.uniform(1.8, 3.1), 150, 6000))
        lot_area = float(np.clip(rng.normal(2.6e6, 9.0e5), 2.0e5, 9.0e6))
        rows.append(
            {
                "geoid": f"36{index:09d}",
                "borough": borough,
                "latitude": round(latitude, 6),
                "longitude": round(longitude, 6),
                "median_gross_rent": round(rent, 2),
                "median_household_income": round(income, 2),
                "population": round(population, 1),
                "housing_units": round(units, 1),
                "assessed_land_value": round(lot_area * rng.uniform(28, 240), 2),
                "lot_area": round(lot_area, 1),
                "residential_area": round(lot_area * rng.uniform(0.25, 1.9), 1),
                "transit_access_score": round(transit, 4),
            }
        )

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(NYC_TRACT_SCHEMA.keys()))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def load_tract_table(path: str | Path) -> Dict[str, np.ndarray]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"tract table not found: {source}")

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        records = list(reader)

    if not records:
        raise ValueError(f"tract table is empty: {source}")

    missing = [c for c in NYC_TRACT_SCHEMA if c not in records[0]]
    if missing:
        raise ValueError(f"tract table {source} is missing columns: {missing}")

    columns: Dict[str, np.ndarray] = {}
    for name in NYC_TRACT_SCHEMA:
        raw = [row[name] for row in records]
        if name in ("geoid", "borough"):
            columns[name] = np.array(raw, dtype=object)
        else:
            columns[name] = np.array([float(value) for value in raw], dtype=np.float64)
    return columns


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    radius = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    inner = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def build_tract_adjacency(
    latitude: np.ndarray,
    longitude: np.ndarray,
    k_nearest: int = 6,
    max_distance_km: Optional[float] = None,
) -> np.ndarray:
    n = latitude.shape[0]
    distances = _haversine_km(
        latitude[:, None], longitude[:, None], latitude[None, :], longitude[None, :]
    )
    np.fill_diagonal(distances, np.inf)

    adjacency = np.zeros((n, n), dtype=np.float64)
    neighbors = np.argsort(distances, axis=1)[:, : max(1, min(k_nearest, n - 1))]
    for node in range(n):
        for neighbor in neighbors[node]:
            if max_distance_km is not None and distances[node, neighbor] > max_distance_km:
                continue
            adjacency[node, neighbor] = 1.0
            adjacency[neighbor, node] = 1.0
    return adjacency


def edge_index_from_adjacency(adjacency: np.ndarray) -> np.ndarray:
    upper = np.triu(adjacency, k=1)
    rows, cols = np.nonzero(upper)
    return np.concatenate(
        [np.stack([rows, cols], axis=0), np.stack([cols, rows], axis=0)], axis=1
    ).astype(np.int64)


def build_panel(
    columns: Dict[str, np.ndarray],
    n_steps: int = 12,
    growth_seed: int = 7,
    annual_rent_growth: float = 0.035,
    annual_income_growth: float = 0.021,
) -> np.ndarray:
    n = columns["median_gross_rent"].shape[0]
    rng = np.random.default_rng(growth_seed)
    panel = np.zeros((n, n_steps, len(PANEL_FEATURE_ORDER)), dtype=np.float64)

    base = np.stack([columns[name] for name in PANEL_FEATURE_ORDER], axis=1)
    panel[:, 0, :] = base

    for step in range(1, n_steps):
        previous = panel[:, step - 1, :]
        noise = rng.normal(0.0, 0.01, size=(n, len(PANEL_FEATURE_ORDER)))
        growth = np.zeros_like(previous)
        growth[:, 0] = annual_rent_growth * 1.4
        growth[:, 1] = annual_rent_growth
        growth[:, 2] = 0.004
        growth[:, 3] = annual_income_growth
        growth[:, 4] = 0.006
        growth[:, 5] = 0.0
        panel[:, step, :] = np.maximum(previous * (1.0 + growth + noise), 1e-6)
    return panel


def to_pyg_data(
    panel: np.ndarray,
    edge_index: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> Any:
    node_features = panel.reshape(panel.shape[0], -1)
    if torch is None or Data is None:
        return {
            "x": node_features,
            "edge_index": edge_index,
            "node_features": panel,
            "feature_names": list(feature_names) if feature_names else None,
        }

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )
    data.num_nodes = int(panel.shape[0])
    data.panel = torch.tensor(panel, dtype=torch.float32)
    if feature_names is not None:
        data.feature_names = list(feature_names)
    return data


def load_empirical_city(
    table_path: Optional[str | Path] = None,
    n_steps: int = 12,
    k_nearest: int = 6,
    max_distance_km: Optional[float] = None,
    allow_sample: bool = True,
    sample_tracts: int = 48,
    seed: int = 42,
) -> Dict[str, Any]:
    is_sample = False
    if table_path is None or not Path(table_path).exists():
        if not allow_sample:
            raise FileNotFoundError(
                "no empirical tract table supplied and allow_sample=False; "
                "provide a CSV matching NYC_TRACT_SCHEMA"
            )
        is_sample = True
        target = Path(table_path) if table_path is not None else Path(
            "./data/empirical/nyc_tract_sample.csv"
        )
        write_sample_tract_table(target, n_tracts=sample_tracts, seed=seed)
        table_path = target

    columns = load_tract_table(table_path)
    adjacency = build_tract_adjacency(
        columns["latitude"],
        columns["longitude"],
        k_nearest=k_nearest,
        max_distance_km=max_distance_km,
    )
    edge_index = edge_index_from_adjacency(adjacency)
    panel = build_panel(columns, n_steps=n_steps, growth_seed=seed)

    return {
        "panel": panel,
        "adjacency": adjacency,
        "edge_index": edge_index,
        "feature_names": list(PANEL_FEATURE_ORDER),
        "geoid": columns["geoid"],
        "borough": columns["borough"],
        "latitude": columns["latitude"],
        "longitude": columns["longitude"],
        "n_neighborhoods": int(panel.shape[0]),
        "n_steps": int(panel.shape[1]),
        "source_path": str(table_path),
        "is_synthetic_sample": is_sample,
        "graph": to_pyg_data(panel, edge_index, PANEL_FEATURE_ORDER),
    }


__all__ = [
    "NYC_TRACT_SCHEMA",
    "PANEL_FEATURE_ORDER",
    "write_sample_tract_table",
    "load_tract_table",
    "build_tract_adjacency",
    "edge_index_from_adjacency",
    "build_panel",
    "to_pyg_data",
    "load_empirical_city",
]
