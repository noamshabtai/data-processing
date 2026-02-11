import copy

import feature_extraction.fft
import feature_extraction.pipeline
import feature_extraction.trend
import numpy as np


def test_trend_extractor_output_shape(kwargs_features):
    kwargs = copy.deepcopy(kwargs_features)
    extractor = feature_extraction.trend.TrendExtractor(**kwargs)
    data = np.random.randn(kwargs["input_size"])
    result = extractor.extract(data)
    assert result.shape[0] > 0


def test_fft_extractor_output_shape(kwargs_features):
    kwargs = copy.deepcopy(kwargs_features)
    extractor = feature_extraction.fft.FFTExtractor(**kwargs)
    data = np.random.randn(kwargs["input_size"])
    result = extractor.extract(data)
    assert result.shape[0] > 0


def test_pipeline_concatenates_all_features(kwargs_features):
    kwargs = copy.deepcopy(kwargs_features)
    pipeline = feature_extraction.pipeline.Pipeline(**kwargs)
    data = np.random.randn(kwargs["input_size"])
    result = pipeline.extract(data)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
