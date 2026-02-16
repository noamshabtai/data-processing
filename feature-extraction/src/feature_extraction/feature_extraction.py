import numpy as np


class FeatureExtraction:
    def __init__(self, **kwargs):
        self.window_size = kwargs.get("window_size", 20)

    def _extract_trend(self, data):
        moving_avg = np.convolve(data, np.ones(self.window_size) / self.window_size, mode="valid")
        slope = np.gradient(moving_avg)
        return np.array([moving_avg[-1], slope[-1], np.std(slope)])

    def _extract_fft(self, data):
        fft_result = np.fft.rfft(data)
        magnitudes = np.abs(fft_result)
        dominant_freq_idx = np.argmax(magnitudes[1:]) + 1
        return np.array([dominant_freq_idx, magnitudes[dominant_freq_idx], np.mean(magnitudes)])

    def execute(self, data):
        features = [self._extract_trend(data), self._extract_fft(data)]
        return np.concatenate(features)
