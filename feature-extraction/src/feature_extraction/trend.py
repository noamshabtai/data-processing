import numpy as np

from . import base


class TrendExtractor(base.BaseExtractor):
    def extract(self, data):
        moving_avg = np.convolve(data, np.ones(self.window_size) / self.window_size, mode="valid")
        slope = np.gradient(moving_avg)
        return np.array([moving_avg[-1], slope[-1], np.std(slope)])
