import copy

import feature_extraction.feature_extraction
import numpy as np
import pytest


def build_signal(simulation):
    match simulation["type"]:
        case "constant":
            return np.full(simulation["size"], simulation["value"])
        case "linear":
            return np.linspace(simulation["start"], simulation["stop"], simulation["size"])
        case "sine":
            return np.sin(2 * np.pi * simulation["cycles"] * np.arange(simulation["size"]) / simulation["size"])


def test_feature_extraction_output_shape(kwargs_features):
    kwargs = copy.deepcopy(kwargs_features)
    tested = feature_extraction.feature_extraction.FeatureExtraction(**kwargs["feature_extraction"])
    data = build_signal(kwargs["simulation"])
    result = tested.execute(data)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == 6


def test_constant_signal_trend(kwargs_features):
    kwargs = copy.deepcopy(kwargs_features)
    if kwargs["simulation"]["type"] != "constant":
        return
    tested = feature_extraction.feature_extraction.FeatureExtraction(**kwargs["feature_extraction"])
    data = build_signal(kwargs["simulation"])
    result = tested.execute(data)
    value = kwargs["simulation"]["value"]
    assert result[0] == pytest.approx(value, abs=1e-10)
    assert result[1] == pytest.approx(0.0, abs=1e-10)
    assert result[2] == pytest.approx(0.0, abs=1e-10)


def test_sine_dominant_frequency(kwargs_features):
    kwargs = copy.deepcopy(kwargs_features)
    if kwargs["simulation"]["type"] != "sine":
        return
    tested = feature_extraction.feature_extraction.FeatureExtraction(**kwargs["feature_extraction"])
    data = build_signal(kwargs["simulation"])
    result = tested.execute(data)
    assert result[3] == kwargs["simulation"]["cycles"]


def test_linear_signal_positive_slope(kwargs_features):
    kwargs = copy.deepcopy(kwargs_features)
    if kwargs["simulation"]["type"] != "linear":
        return
    tested = feature_extraction.feature_extraction.FeatureExtraction(**kwargs["feature_extraction"])
    data = build_signal(kwargs["simulation"])
    result = tested.execute(data)
    assert result[1] > 0
