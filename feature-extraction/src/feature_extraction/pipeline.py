import numpy as np

from . import fft, trend


class Pipeline:
    def __init__(self, **kwargs):
        self.extractors = [
            trend.TrendExtractor(**kwargs),
            fft.FFTExtractor(**kwargs),
        ]

    def extract(self, data):
        features = [extractor.extract(data) for extractor in self.extractors]
        return np.concatenate(features)
