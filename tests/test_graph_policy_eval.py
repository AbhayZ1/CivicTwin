import numpy as np

from civictwin.evaluate import affordability, compare_policies, displacement_pressure_index
from civictwin.graph import build_graph
from civictwin.policy import (
    Policy,
    apply_policy,
    baseline_policy,
    inclusionary_housing_policy,
    market_led_policy,
)


def test_build_graph_uses_isolated_nodes_when_no_edges():
    panel = np.ones((4, 3, 2), dtype=np.float32)
    graph = build_graph(panel, edge_index=np.array([[0, 1], [1, 0]]), no_edges=True)

    assert graph["x"].shape == (4, 6)
    assert graph["edge_index"].shape == (2, 0)


def test_policy_application_changes_selected_nodes_only():
    panel = np.ones((4, 5, 6), dtype=np.float64)
    policy = market_led_policy(boundary=[1, 2], timing=1, intensity=0.5)
    updated = apply_policy(panel, policy)

    assert updated[0, 1, 0] == panel[0, 1, 0]
    assert updated[1, 1, 0] > panel[1, 1, 0]
    assert updated[2, 1, 0] > panel[2, 1, 0]


def test_baseline_policy_is_noop_and_eval_works():
    panel = np.array(
        [
            [
                [100.0, 50.0, 2000.0, 60000.0, 900.0, 1.0],
                [110.0, 55.0, 2050.0, 61000.0, 920.0, 1.1],
            ],
            [
                [90.0, 40.0, 1900.0, 50000.0, 850.0, 1.0],
                [95.0, 42.0, 1920.0, 49500.0, 840.0, 1.0],
            ],
        ],
        dtype=np.float64,
    )
    assert baseline_policy().name == "baseline"
    assert affordability(panel).shape == (2, 2)
    assert displacement_pressure_index(panel).shape == (2,)

    policies = [
        Policy(name="baseline", boundary=[], timing=0, intensity=0.0),
        inclusionary_housing_policy(boundary=[0], timing=0, intensity=1.0),
    ]
    frame = compare_policies(panel, policies)
    assert set(frame["policy"]) == {"baseline", "inclusionary_housing"}
    assert "ranking" in frame.columns
