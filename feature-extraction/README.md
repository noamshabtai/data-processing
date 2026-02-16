# feature-extraction

Time-series feature extraction using trend analysis and FFT.

## FeatureExtraction

Single class that extracts trend and frequency features from a data window.

**Output (6 values):** `[last_moving_avg, last_slope, slope_std, dominant_freq_index, dominant_magnitude, mean_magnitude]`

- `_extract_trend` — moving average, slope, and slope volatility (3 values)
- `_extract_fft` — dominant frequency index, its magnitude, and mean magnitude (3 values)

## Usage

```python
from feature_extraction.feature_extraction import FeatureExtraction

fe = FeatureExtraction(window_size=20)
features = fe.execute(data)  # numpy array of 6 values
```

## Dependencies

- numpy
- scipy
