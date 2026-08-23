import numpy as np

from civictwin.synth import generate_synthetic_city


def test_synth_is_reproducible_and_shapes_are_expected():
    city_a = generate_synthetic_city(seed=7, n_neighborhoods=9, n_steps=10)
    city_b = generate_synthetic_city(seed=7, n_neighborhoods=9, n_steps=10)

    assert np.allclose(city_a["panel"], city_b["panel"])
    assert city_a["panel"].shape == (9, 10, 6)
    assert city_a["adjacency"].shape == (9, 9)
    assert city_a["edge_index"].shape[0] == 2

    # Land value should increase after the accessibility shock.
    shock_step = 3
    assert city_a["panel"][0, shock_step + 1, 0] >= city_a["panel"][0, shock_step, 0]
