import copy

import torch

import model.model


def test_forward(kwargs_model):
    kwargs = copy.deepcopy(kwargs_model)
    tested = model.model.Model(**kwargs["network"])
    batch_size = kwargs["simulation"]["batch_size"]
    seq_len = kwargs["simulation"]["seq_len"]
    x = torch.randn(batch_size, seq_len, kwargs["network"]["input_dim"])
    output = tested(x)
    assert output.shape == (batch_size, kwargs["network"]["output_dim"])


def test_load_weights(kwargs_model, tmp_path):
    kwargs = copy.deepcopy(kwargs_model)
    source = model.model.Model(**kwargs["network"])
    path = tmp_path / "weights.pt"
    torch.save(source.state_dict(), path)
    tested = model.model.Model(**kwargs["network"])
    tested.load_weights(path)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(tested.state_dict()[key], value)
