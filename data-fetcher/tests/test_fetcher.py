import copy

import data_fetcher.fetcher
import pandas as pd


def test_fetcher_fetch_historical_returns_dataframe(kwargs_fetcher):
    kwargs = copy.deepcopy(kwargs_fetcher)
    fetcher = data_fetcher.fetcher.Fetcher(**kwargs)
    df = fetcher.fetch_historical()
    assert isinstance(df, pd.DataFrame)


def test_fetcher_fetch_historical_has_ohlcv_columns(kwargs_fetcher):
    kwargs = copy.deepcopy(kwargs_fetcher)
    fetcher = data_fetcher.fetcher.Fetcher(**kwargs)
    df = fetcher.fetch_historical()
    expected_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in expected_columns:
        assert col in df.columns


def test_fetcher_fetch_latest_returns_series(kwargs_fetcher):
    kwargs = copy.deepcopy(kwargs_fetcher)
    fetcher = data_fetcher.fetcher.Fetcher(**kwargs)
    latest = fetcher.fetch_latest()
    assert isinstance(latest, pd.Series)
