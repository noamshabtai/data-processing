import feature_extraction.fft
import feature_extraction.pipeline
import feature_extraction.trend
import numpy as np


def test_trend_extractor_output_shape(kwargs_features):
    extractor = feature_extraction.trend.TrendExtractor(**kwargs_features)
    data = np.random.randn(kwargs_features["input_size"])
    result = extractor.extract(data)
    assert result.shape[0] > 0


def test_fft_extractor_output_shape(kwargs_features):
    extractor = feature_extraction.fft.FFTExtractor(**kwargs_features)
    data = np.random.randn(kwargs_features["input_size"])
    result = extractor.extract(data)
    assert result.shape[0] > 0


def test_pipeline_concatenates_all_features(kwargs_features):
    pipeline = feature_extraction.pipeline.Pipeline(**kwargs_features)
    data = np.random.randn(kwargs_features["input_size"])
    result = pipeline.extract(data)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
