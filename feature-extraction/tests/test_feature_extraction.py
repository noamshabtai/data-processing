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


def check_shape(result):
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert len(result) == 6


def check_constant(result, simulation):
    assert result[0] == pytest.approx(simulation["value"], abs=1e-10)
    assert result[1] == pytest.approx(0.0, abs=1e-10)
    assert result[2] == pytest.approx(0.0, abs=1e-10)


def check_linear(result):
    assert result[1] > 0


def check_sine(result, simulation):
    assert result[3] == simulation["cycles"]


def test_execute(kwargs_feature_extraction_execute):
    kwargs = copy.deepcopy(kwargs_feature_extraction_execute)
    tested = feature_extraction.feature_extraction.FeatureExtraction(**kwargs["feature_extraction"])
    data = build_signal(kwargs["simulation"])
    result = tested.execute(data)

    check_shape(result)
    match kwargs["simulation"]["type"]:
        case "constant":
            check_constant(result, kwargs["simulation"])
        case "linear":
            check_linear(result)
        case "sine":
            check_sine(result, kwargs["simulation"])
