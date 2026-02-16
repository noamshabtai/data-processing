import copy

import numpy as np

import model.model


def test_execute_output_shape(kwargs_model):
    kwargs = copy.deepcopy(kwargs_model)
    m = model.model.Model(**kwargs["model"])
    seq_len = kwargs["simulation"]["seq_len"]
    features = np.random.randn(seq_len, kwargs["model"]["input_dim"]).astype(np.float32)
    result = m.execute(features)
    assert result.shape == (kwargs["model"]["output_dim"],)


def test_execute_output_type(kwargs_model):
    kwargs = copy.deepcopy(kwargs_model)
    m = model.model.Model(**kwargs["model"])
    seq_len = kwargs["simulation"]["seq_len"]
    features = np.random.randn(seq_len, kwargs["model"]["input_dim"]).astype(np.float32)
    result = m.execute(features)
    assert isinstance(result, np.ndarray)


def test_backward_reduces_loss(kwargs_model):
    kwargs = copy.deepcopy(kwargs_model)
    m = model.model.Model(**kwargs["model"])
    batch_size = kwargs["simulation"]["batch_size"]
    seq_len = kwargs["simulation"]["seq_len"]
    data = np.random.randn(batch_size, seq_len, kwargs["model"]["input_dim"]).astype(np.float32)
    targets = np.random.randn(batch_size, kwargs["model"]["output_dim"]).astype(np.float32)
    loss_before = m.backward(data, targets, epochs=1, lr=kwargs["training"]["lr"])
    loss_after = m.backward(data, targets, epochs=kwargs["training"]["epochs"], lr=kwargs["training"]["lr"])
    assert loss_after < loss_before


def test_save_and_load(kwargs_model, tmp_path):
    kwargs = copy.deepcopy(kwargs_model)
    m = model.model.Model(**kwargs["model"])
    seq_len = kwargs["simulation"]["seq_len"]
    features = np.random.randn(seq_len, kwargs["model"]["input_dim"]).astype(np.float32)
    result_before = m.execute(features)
    path = tmp_path / "model.pt"
    m.save(path)
    m2 = model.model.Model(**kwargs["model"])
    m2.load(path)
    result_after = m2.execute(features)
    np.testing.assert_array_equal(result_before, result_after)
