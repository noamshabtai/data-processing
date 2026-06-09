import copy

import feature_extraction.feature_extraction
import numpy as np
import stock_analyzer.predictor
import torch

import model.model


def test_init(kwargs_predictor):
    kwargs = copy.deepcopy(kwargs_predictor)
    tested = stock_analyzer.predictor.System(**kwargs["predictor"])
    assert isinstance(tested.modules["feature_extraction"], feature_extraction.feature_extraction.FeatureExtraction)
    assert isinstance(tested.modules["model"], model.model.Model)


def test_connect(kwargs_predictor):
    kwargs = copy.deepcopy(kwargs_predictor)
    tested = stock_analyzer.predictor.System(**kwargs["predictor"])
    tested.connect("feature_extraction")
    assert tested.inputs["feature_extraction"]["data"] is tested.input_buffer.buffer
    tested.outputs["feature_extraction"] = tested.modules["feature_extraction"].execute(
        **tested.inputs["feature_extraction"]
    )
    tested.connect("model")
    x = tested.inputs["model"]["x"]
    input_dim = kwargs["predictor"]["model"]["input_dim"]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, 1, input_dim)
    assert x.dtype == torch.float32


def test_execute(kwargs_predictor):
    kwargs = copy.deepcopy(kwargs_predictor)
    tested = stock_analyzer.predictor.System(**kwargs["predictor"])
    for _ in range(tested.input_buffer.steps_to_ready):
        tested.execute(np.zeros(1, dtype=np.float32))
    assert "model" in tested.outputs
