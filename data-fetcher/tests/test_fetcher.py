import copy

import data_fetcher.fetcher
import pandas as pd

MOCK_HISTORICAL_DF = pd.DataFrame(
    {"Open": [1.0, 2.0], "High": [1.5, 2.5], "Low": [0.5, 1.5], "Close": [1.2, 2.2], "Volume": [100, 200]}
)


def test_fetcher_fetch_historical_returns_dataframe(mocker, kwargs_fetcher):
    kwargs = copy.deepcopy(kwargs_fetcher)
    mock_ticker = mocker.patch("yfinance.Ticker")
    mock_ticker.return_value.history.return_value = MOCK_HISTORICAL_DF
    fetcher = data_fetcher.fetcher.Fetcher(**kwargs)
    df = fetcher.fetch_historical()
    assert isinstance(df, pd.DataFrame)


def test_fetcher_fetch_historical_has_ohlcv_columns(mocker, kwargs_fetcher):
    kwargs = copy.deepcopy(kwargs_fetcher)
    mock_ticker = mocker.patch("yfinance.Ticker")
    mock_ticker.return_value.history.return_value = MOCK_HISTORICAL_DF
    fetcher = data_fetcher.fetcher.Fetcher(**kwargs)
    df = fetcher.fetch_historical()
    expected_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in expected_columns:
        assert col in df.columns


def test_fetcher_fetch_latest_returns_series(mocker, kwargs_fetcher):
    kwargs = copy.deepcopy(kwargs_fetcher)
    mock_ticker = mocker.patch("yfinance.Ticker")
    mock_ticker.return_value.history.return_value = MOCK_HISTORICAL_DF
    fetcher = data_fetcher.fetcher.Fetcher(**kwargs)
    latest = fetcher.fetch_latest()
    assert isinstance(latest, pd.Series)
