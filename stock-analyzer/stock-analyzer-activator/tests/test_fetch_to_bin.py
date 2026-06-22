import copy

import numpy as np
import pandas as pd
import stock_analyzer_activator.fetch_to_bin

MOCK_HISTORICAL_DF = pd.DataFrame(
    {"Open": [1.0, 2.0], "High": [1.5, 2.5], "Low": [0.5, 1.5], "Close": [1.2, 2.2], "Volume": [100, 200]}
)


def test_fetch_to_bin(mocker, kwargs_fetch_to_bin):
    kwargs = copy.deepcopy(kwargs_fetch_to_bin)
    mock_ticker = mocker.patch("yfinance.Ticker")
    mock_ticker.return_value.history.return_value = MOCK_HISTORICAL_DF
    stock_analyzer_activator.fetch_to_bin.fetch_to_bin(**kwargs)
    written = np.fromfile(kwargs["output_path"], dtype=np.float32)
    expected = MOCK_HISTORICAL_DF[kwargs["field"]].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(written, expected)
