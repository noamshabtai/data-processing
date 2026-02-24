import copy

import torch

import model.model


def test_forward_output_shape(kwargs_model):
    kwargs = copy.deepcopy(kwargs_model)
    tested = model.model.Model(**kwargs["network"])
    batch_size = kwargs["simulation"]["batch_size"]
    seq_len = kwargs["simulation"]["seq_len"]
    x = torch.randn(batch_size, seq_len, kwargs["network"]["input_dim"])
    output = tested(x)
    assert output.shape == (batch_size, kwargs["network"]["output_dim"])


def test_forward_output_type(kwargs_model):
    kwargs = copy.deepcopy(kwargs_model)
    tested = model.model.Model(**kwargs["network"])
    batch_size = kwargs["simulation"]["batch_size"]
    seq_len = kwargs["simulation"]["seq_len"]
    x = torch.randn(batch_size, seq_len, kwargs["network"]["input_dim"])
    output = tested(x)
    assert isinstance(output, torch.Tensor)


def test_save_and_load(kwargs_model, tmp_path):
    kwargs = copy.deepcopy(kwargs_model)
    tested = model.model.Model(**kwargs["network"])
    batch_size = kwargs["simulation"]["batch_size"]
    seq_len = kwargs["simulation"]["seq_len"]
    x = torch.randn(batch_size, seq_len, kwargs["network"]["input_dim"])
    result_before = tested(x)
    path = tmp_path / "model.pt"
    torch.save(tested.state_dict(), path)
    loaded = model.model.Model(**kwargs["network"])
    loaded.load(path)
    result_after = loaded(x)
    torch.testing.assert_close(result_before, result_after)
