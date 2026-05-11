import copy

import numpy as np
import torch

import model.trainer


def test_execute(kwargs_trainer):
    kwargs = copy.deepcopy(kwargs_trainer)
    tested = model.trainer.Trainer(**kwargs["model"])
    seq_len = kwargs["simulation"]["seq_len"]
    features = np.random.randn(seq_len, kwargs["model"]["network"]["input_dim"]).astype(np.float32)
    result = tested.execute(features)
    assert result.shape == (kwargs["model"]["network"]["output_dim"],)


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


def test_save_checkpoint(kwargs_trainer, tmp_path):
    kwargs = copy.deepcopy(kwargs_trainer)
    tested = model.trainer.Trainer(**kwargs["model"])
    path = tmp_path / "checkpoint.pt"
    tested.save_checkpoint(path)
    saved = torch.load(path, weights_only=True)
    assert saved["model"].keys() == tested.model.state_dict().keys()
    assert saved["optimizer"].keys() == tested.optimizer.state_dict().keys()


def test_load_checkpoint(kwargs_trainer, tmp_path):
    kwargs = copy.deepcopy(kwargs_trainer)
    source = model.trainer.Trainer(**kwargs["model"])
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": source.model.state_dict(), "optimizer": source.optimizer.state_dict()}, path)
    tested = model.trainer.Trainer(**kwargs["model"])
    tested.load_checkpoint(path)
    for key, value in source.model.state_dict().items():
        torch.testing.assert_close(tested.model.state_dict()[key], value)
