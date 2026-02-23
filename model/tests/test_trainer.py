import copy

import numpy as np

import model.trainer


def test_execute_output_shape(kwargs_trainer):
    kwargs = copy.deepcopy(kwargs_trainer)
    tested = model.trainer.Trainer(**kwargs["model"])
    seq_len = kwargs["simulation"]["seq_len"]
    features = np.random.randn(seq_len, kwargs["model"]["network"]["input_dim"]).astype(np.float32)
    result = tested.execute(features)
    assert result.shape == (kwargs["model"]["network"]["output_dim"],)


def test_execute_output_type(kwargs_trainer):
    kwargs = copy.deepcopy(kwargs_trainer)
    tested = model.trainer.Trainer(**kwargs["model"])
    seq_len = kwargs["simulation"]["seq_len"]
    features = np.random.randn(seq_len, kwargs["model"]["network"]["input_dim"]).astype(np.float32)
    result = tested.execute(features)
    assert isinstance(result, np.ndarray)


def test_backward_reduces_loss(kwargs_trainer):
    kwargs = copy.deepcopy(kwargs_trainer)
    tested = model.trainer.Trainer(**{**kwargs["model"], "data_shuffle": False})
    num_samples = kwargs["simulation"]["num_samples"]
    seq_len = kwargs["simulation"]["seq_len"]
    np.random.seed(42)
    data = np.random.randn(num_samples, seq_len, kwargs["model"]["network"]["input_dim"]).astype(np.float32)
    targets = np.random.randn(num_samples, kwargs["model"]["network"]["output_dim"]).astype(np.float32)
    epoch_losses = tested.backward(data, targets)
    assert np.all(np.diff(epoch_losses) <= 0)


def test_save_and_load(kwargs_trainer, tmp_path):
    kwargs = copy.deepcopy(kwargs_trainer)
    tested = model.trainer.Trainer(**kwargs["model"])
    seq_len = kwargs["simulation"]["seq_len"]
    features = np.random.randn(seq_len, kwargs["model"]["network"]["input_dim"]).astype(np.float32)
    result_before = tested.execute(features)
    path = tmp_path / "model.pt"
    tested.save(path)
    loaded = model.trainer.Trainer(**kwargs["model"])
    loaded.load(path)
    result_after = loaded.execute(features)
    np.testing.assert_array_equal(result_before, result_after)
