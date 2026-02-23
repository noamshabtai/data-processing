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
