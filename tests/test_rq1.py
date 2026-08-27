import numpy as np
import torch

from civictwin.experiments.run_rq1 import (
    build_policy_scorecard,
    evaluate_model,
    make_windows,
    run_forecast_experiment,
    train_model,
)
from civictwin.model import MLPBaseline, SpatioTemporalGNN
from civictwin.synth import generate_synthetic_city


def test_graph_model_beats_baseline_on_spillover_features():
    city = generate_synthetic_city(seed=7, n_neighborhoods=9, n_steps=18)
    panel = city["panel"][:, :, [0, 1, 5]]
    edge_index = torch.tensor(city["edge_index"], dtype=torch.long)
    feature_names = ["land_value", "rent", "accessibility"]

    windows, targets = make_windows(panel, window_size=4)
    split = max(1, int(len(windows) * 0.7))
    train_windows, train_targets = windows[:split], targets[:split]
    test_windows, test_targets = windows[split:], targets[split:]

    graph_model = SpatioTemporalGNN(input_dim=3, hidden_dim=12, output_dim=3)
    mlp_model = MLPBaseline(input_dim=3, hidden_dim=12, output_dim=3, sequence_length=4)

    train_model(graph_model, train_windows, train_targets, edge_index, device=torch.device("cpu"), epochs=150, learning_rate=1e-3)
    train_model(mlp_model, train_windows, train_targets, edge_index, device=torch.device("cpu"), epochs=150, learning_rate=1e-3)

    graph_metrics = evaluate_model(graph_model, test_windows, test_targets, edge_index, feature_names, torch.device("cpu"))
    mlp_metrics = evaluate_model(mlp_model, test_windows, test_targets, edge_index, feature_names, torch.device("cpu"))

    assert graph_metrics["overall_mae"] < mlp_metrics["overall_mae"]


def test_policy_scorecard_includes_configured_scenarios():
    city = generate_synthetic_city(seed=3, n_neighborhoods=6, n_steps=8)

    scorecard = build_policy_scorecard(city)

    assert set(scorecard["policy"]) == {"baseline", "market_led", "inclusionary_housing", "land_value_capture"}
    assert scorecard["ranking"].notna().all()


def test_forecast_experiment_returns_seeded_result_row():
    city = generate_synthetic_city(seed=3, n_neighborhoods=6, n_steps=8)

    result = run_forecast_experiment(
        seed=3,
        n_neighborhoods=6,
        n_steps=8,
        epochs=1,
        config=city["config"],
    )

    assert result["seed"] == 3
    assert result["graph_overall_mae"] >= 0
    assert result["baseline_overall_mae"] >= 0
