# feature-extraction

Time-series feature extraction using trend analysis and FFT.

## Extractors

### TrendExtractor

Computes moving average, slope, and slope volatility from a data window.

**Output (3 values):** `[last_moving_avg, last_slope, slope_std]`

### FFTExtractor

Performs Fast Fourier Transform to identify dominant frequency components.

**Output (3 values):** `[dominant_freq_index, dominant_magnitude, mean_magnitude]`

### Pipeline

Combines both extractors into a single feature vector.

**Output (6 values):** concatenation of TrendExtractor and FFTExtractor outputs.

## Usage

```python
from feature_extraction.pipeline import Pipeline

pipeline = Pipeline(window_size=20)
features = pipeline.extract(data)  # numpy array of 6 values
```

## Dependencies

- numpy
- scipy
