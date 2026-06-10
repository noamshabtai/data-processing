import copy

import feature_extraction.feature_extraction
import numpy as np
import stock_analyzer.trainer

import model.trainer


def test_init(kwargs_trainer):
    kwargs = copy.deepcopy(kwargs_trainer)
    tested = stock_analyzer.trainer.System(**kwargs["trainer"])
    assert isinstance(tested.modules["feature_extraction"], feature_extraction.feature_extraction.FeatureExtraction)
    assert isinstance(tested.modules["trainer"], model.trainer.Trainer)


def test_connect(kwargs_trainer):
    kwargs = copy.deepcopy(kwargs_trainer)
    tested = stock_analyzer.trainer.System(**kwargs["trainer"])
    tested.connect("feature_extraction")
    assert tested.inputs["feature_extraction"]["data"] is tested.input_buffer.buffer
    tested.outputs["feature_extraction"] = tested.modules["feature_extraction"].execute(
        **tested.inputs["feature_extraction"]
    )
    tested.connect("trainer")
    features = tested.inputs["trainer"]["features"]
    input_dim = kwargs["trainer"]["trainer"]["network"]["input_dim"]
    assert isinstance(features, np.ndarray)
    assert features.shape == (1, input_dim)
    assert features.dtype == np.float32


def test_execute(kwargs_trainer):
    kwargs = copy.deepcopy(kwargs_trainer)
    tested = stock_analyzer.trainer.System(**kwargs["trainer"])
    for _ in range(tested.input_buffer.steps_to_ready):
        tested.execute(np.zeros(1, dtype=np.float32))
    assert "trainer" in tested.outputs
