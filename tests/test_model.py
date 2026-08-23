import numpy as np
import torch

from civictwin.model import MLPBaseline, SpatioTemporalGNN


def test_model_forward_shapes_are_valid():
    x = torch.randn(4, 3, 6)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    model = SpatioTemporalGNN(input_dim=6, hidden_dim=8, output_dim=6)
    outputs = model(x, edge_index)
    assert outputs.shape == (4, 6)

    baseline = MLPBaseline(input_dim=6, hidden_dim=8, output_dim=6, sequence_length=3)
    base_out = baseline(x)
    assert base_out.shape == (4, 6)
