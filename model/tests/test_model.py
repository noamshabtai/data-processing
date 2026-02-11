import copy

import numpy as np
import torch

import model.lstm
import model.predictor


def test_lstm_forward_shape(kwargs_model):
    kwargs = copy.deepcopy(kwargs_model)
    lstm = model.lstm.LSTM(**kwargs)
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, kwargs["input_dim"])
    output = lstm(x)
    assert output.shape == (batch_size, kwargs["output_dim"])


def test_predictor_returns_numpy(kwargs_model):
    kwargs = copy.deepcopy(kwargs_model)
    predictor = model.predictor.Predictor(**kwargs)
    features = np.random.randn(10, kwargs["input_dim"]).astype(np.float32)
    result = predictor.predict(features)
    assert isinstance(result, np.ndarray)
