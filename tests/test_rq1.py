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


def test_learned_models_beat_naive_persistence_on_scaled_pipeline():
    for seed in (3, 7):
        result = run_forecast_experiment(
            seed=seed, n_neighborhoods=9, n_steps=18, epochs=200, device="cpu"
        )
        assert result["stgnn_overall_mae"] < result["persistence_overall_mae"]
        assert result["slm_overall_mae"] < result["persistence_overall_mae"]
        assert result["mlp_overall_mae"] < result["persistence_overall_mae"]


def test_stgnn_training_reduces_error_versus_untrained_model():
    city = generate_synthetic_city(seed=7, n_neighborhoods=9, n_steps=18)
    panel = city["panel"][:, :, [0, 1, 5]]
    edge_index = torch.tensor(city["edge_index"], dtype=torch.long)
    feature_names = ["land_value", "rent", "accessibility"]
    device = torch.device("cpu")

    windows, targets = make_windows(panel, window_size=4)
    split = max(1, int(len(windows) * 0.7))
    train_windows, train_targets = windows[:split], targets[:split]
    test_windows, test_targets = windows[split:], targets[split:]

    torch.manual_seed(0)
    untrained = SpatioTemporalGNN(input_dim=3, hidden_dim=12, output_dim=3)
    before = evaluate_model(
        untrained, test_windows, test_targets, edge_index, feature_names, device
    )

    torch.manual_seed(0)
    trained = SpatioTemporalGNN(input_dim=3, hidden_dim=12, output_dim=3)
    train_model(
        trained,
        train_windows,
        train_targets,
        edge_index,
        device=device,
        epochs=150,
        learning_rate=1e-3,
    )
    after = evaluate_model(
        trained, test_windows, test_targets, edge_index, feature_names, device
    )

    assert after["overall_mae"] < before["overall_mae"]


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
