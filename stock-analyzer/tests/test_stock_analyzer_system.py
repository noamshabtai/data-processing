import copy

import feature_extraction.feature_extraction
import stock_analyzer.system

import model.trainer


def test_init(kwargs_system):
    kwargs = copy.deepcopy(kwargs_system)
    tested = stock_analyzer.system.System(**kwargs["system"])
    assert isinstance(tested.modules["feature_extraction"], feature_extraction.feature_extraction.FeatureExtraction)
    assert isinstance(tested.modules["trainer"], model.trainer.Trainer)


def test_connect(kwargs_system):
    kwargs = copy.deepcopy(kwargs_system)
    tested = stock_analyzer.system.System(**kwargs["system"])
    tested.connect("feature_extraction")
    assert tested.inputs["feature_extraction"]["data"] is tested.input_buffer.buffer
    tested.outputs["feature_extraction"] = tested.modules["feature_extraction"].execute(
        **tested.inputs["feature_extraction"]
    )
    tested.connect("trainer")
    assert tested.inputs["trainer"]["features"] is tested.outputs["feature_extraction"]
