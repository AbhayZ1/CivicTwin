import numpy as np
import torch

from civictwin.data.empirical_loader import (
    NYC_TRACT_SCHEMA,
    build_tract_adjacency,
    load_empirical_city,
)
from civictwin.evaluate import (
    cumulative_rent_burden,
    displacement_pressure_index,
    housing_transport_index,
    mae,
    regression_metrics,
    rmse,
    transit_cost,
)
from civictwin.model import (
    MLPBaseline,
    PersistenceBaseline,
    SpatialLagModel,
    SpatioTemporalGNN,
)
from civictwin.paper.latex_export import build_algorithm_block
from civictwin.synth import scenario_panels


def _panel():
    return np.array(
        [
            [
                [100.0, 50.0, 2000.0, 60000.0, 900.0, 1.0],
                [110.0, 55.0, 2050.0, 61000.0, 920.0, 1.5],
                [120.0, 60.0, 2100.0, 62000.0, 940.0, 2.0],
            ],
            [
                [90.0, 40.0, 1900.0, 50000.0, 850.0, 1.0],
                [95.0, 42.0, 1920.0, 49500.0, 840.0, 0.9],
                [99.0, 44.0, 1930.0, 49000.0, 830.0, 0.8],
            ],
        ],
        dtype=np.float64,
    )


def test_ht_index_matches_specification_formula():
    panel = _panel()
    params = {"transit_base_cost": 20.0, "transit_accessibility_elasticity": 1.0}

    index = housing_transport_index(panel, params)
    expected = (panel[:, :, 1] + 20.0 / panel[:, :, 5]) / panel[:, :, 3]

    assert index.shape == (2, 3)
    assert np.allclose(index, expected)
    assert np.allclose(transit_cost(panel, params), 20.0 / panel[:, :, 5])


def test_dpi_reduces_to_specification_formula_when_delta_is_zero():
    panel = _panel()
    share = np.array([[0.50, 0.45, 0.40], [0.60, 0.62, 0.64]])
    weights = {"alpha": 0.4, "beta": 0.3, "gamma": 0.3, "delta": 0.0}

    result = displacement_pressure_index(
        panel, weights=weights, low_income_share=share, per_step=True
    )

    rent = panel[:, :, 1]
    accessibility = panel[:, :, 5]
    expected = (
        0.4 * (np.diff(rent, axis=1) / rent[:, :-1])
        + 0.3 * (1.0 - share[:, 1:] / share[:, :-1])
        + 0.3 * np.diff(accessibility, axis=1)
    )

    assert result.shape == (2, 2)
    assert np.allclose(result, expected)
    assert displacement_pressure_index(panel, weights, share).shape == (2,)


def test_dpi_level_term_adds_cumulative_rent_burden():
    panel = _panel()
    share = np.array([[0.50, 0.45, 0.40], [0.60, 0.62, 0.64]])
    weights = {"alpha": 0.4, "beta": 0.3, "gamma": 0.3, "delta": 0.5}

    with_level = displacement_pressure_index(
        panel, weights=weights, low_income_share=share, per_step=True
    )
    without_level = displacement_pressure_index(
        panel, weights={**weights, "delta": 0.0}, low_income_share=share, per_step=True
    )
    burden = cumulative_rent_burden(panel)

    assert np.allclose(burden[:, 0], 0.0)
    assert np.allclose(with_level - without_level, 0.5 * burden[:, 1:])


def test_dpi_keeps_permanent_policy_shift_visible_at_long_horizon():
    city = scenario_panels(seed=1, n_neighborhoods=9, n_steps=40)
    baseline = city["scenarios"]["baseline"]
    market = city["scenarios"]["market_led"]

    late = slice(-8, None)
    delta = (
        displacement_pressure_index(market, per_step=True).mean(axis=0)[late]
        - displacement_pressure_index(baseline, per_step=True).mean(axis=0)[late]
    )

    assert np.all(delta > 1e-3)


def test_rmse_is_a_true_root_mean_squared_error():
    targets = np.zeros((2, 3, 2))
    predictions = np.zeros_like(targets)
    predictions[0, 0, 0] = 3.0
    predictions[0, 0, 1] = 4.0

    metrics = regression_metrics(predictions, targets, ["a", "b"])

    assert np.isclose(metrics["a_rmse"], np.sqrt(9.0 / 6.0))
    assert np.isclose(metrics["a_mae"], 0.5)
    assert np.isclose(metrics["overall_rmse"], np.sqrt(25.0 / 12.0))
    assert metrics["overall_rmse"] >= metrics["overall_mae"]
    assert np.isclose(rmse(predictions, targets), metrics["overall_rmse"])
    assert np.isclose(mae(predictions, targets), metrics["overall_mae"])


def test_stgnn_decomposes_into_direct_and_spillover_terms():
    torch.manual_seed(0)
    x = torch.randn(4, 3, 6)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    treatment = torch.tensor([1.0, 0.0, 0.0, 0.0])

    model = SpatioTemporalGNN(input_dim=6, hidden_dim=8, output_dim=6)
    parts = model.decompose(x, edge_index, treatment)

    assert torch.allclose(parts.prediction, parts.direct + parts.spillover, atol=1e-6)

    effects = model.causal_effects(x, edge_index, treatment)
    assert effects["ite"][1:].abs().max().item() == 0.0
    assert effects["ite"][0].abs().max().item() > 0.0
    assert effects["ste"][1].abs().max().item() > 0.0

    isolated = model.decompose(x, torch.zeros(2, 0, dtype=torch.long), treatment)
    assert isolated.spillover.abs().max().item() == 0.0


def test_baseline_family_shapes_and_graph_dependence():
    torch.manual_seed(0)
    x = torch.randn(4, 3, 6)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    assert MLPBaseline(6, 8, 6, 3)(x).shape == (4, 6)
    assert PersistenceBaseline()(x).shape == (4, 6)
    assert torch.allclose(PersistenceBaseline()(x), x[:, -1, :])

    slm = SpatialLagModel(input_dim=6, output_dim=6, sequence_length=3)
    with torch.no_grad():
        slm.rho.fill_(0.5)
    linked = slm(x, edge_index)
    isolated = slm(x, torch.zeros(2, 0, dtype=torch.long))

    assert linked.shape == (4, 6)
    assert not torch.allclose(linked, isolated)


def test_scenario_panels_cover_all_four_policies():
    city = scenario_panels(seed=3, n_neighborhoods=9, n_steps=8)

    assert set(city["scenarios"]) == {
        "baseline",
        "market_led",
        "inclusionary_housing",
        "land_value_capture",
    }
    baseline = city["scenarios"]["baseline"]
    market = city["scenarios"]["market_led"]
    assert market[:, :, 1].mean() > baseline[:, :, 1].mean()
    assert city["scenarios"]["inclusionary_housing"][:, :, 1].mean() < baseline[:, :, 1].mean()


def test_empirical_loader_builds_graph_from_tract_table(tmp_path):
    city = load_empirical_city(
        table_path=tmp_path / "tracts.csv", n_steps=6, k_nearest=4, sample_tracts=20
    )

    assert city["is_synthetic_sample"] is True
    assert city["panel"].shape == (20, 6, 6)
    assert city["edge_index"].shape[0] == 2
    assert city["edge_index"].shape[1] > 0
    assert set(NYC_TRACT_SCHEMA).issuperset({"geoid", "median_gross_rent", "latitude"})

    adjacency = build_tract_adjacency(city["latitude"], city["longitude"], k_nearest=4)
    assert np.allclose(adjacency, adjacency.T)
    assert np.all(np.diag(adjacency) == 0.0)


def test_algorithm_block_is_wellformed_latex():
    block = build_algorithm_block()

    assert block.count(r"\begin{algorithm}") == 1
    assert block.count(r"\end{algorithm}") == 1
    assert r"\begin{algorithmic}" in block
    assert r"\Phi" in block and r"\Psi" in block
    assert "ITE" in block and "STE" in block
