import numpy as np

from . import base


class FFTExtractor(base.BaseExtractor):
    def extract(self, data):
        fft_result = np.fft.rfft(data)
        magnitudes = np.abs(fft_result)
        dominant_freq_idx = np.argmax(magnitudes[1:]) + 1
        return np.array([dominant_freq_idx, magnitudes[dominant_freq_idx], np.mean(magnitudes)])
