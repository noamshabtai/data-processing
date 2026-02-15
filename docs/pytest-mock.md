# Mocking External Dependencies with pytest-mock

How the `data-fetcher` tests use `pytest-mock` to avoid hitting Yahoo Finance during testing.

## The Problem

`Fetcher` calls `yfinance.Ticker` under the hood, which reaches out to Yahoo's servers. Tests that depend on network calls are slow, flaky, and break offline. Mocking replaces the real dependency with a fake that returns controlled data.

## How `mocker.patch` Works

The `pytest-mock` plugin provides a `mocker` fixture — a thin wrapper around Python's `unittest.mock`. Add it as a test parameter and pytest injects it automatically.

```python
def test_something(mocker):
    mock_ticker = mocker.patch("yfinance.Ticker")
```

`mocker.patch("yfinance.Ticker")` replaces the real `yfinance.Ticker` with a `MagicMock` object for the duration of the test. When the test ends, the original is automatically restored.

Python modules are just objects with attributes — a class is simply an attribute on a module. The patch essentially does:

```python
yfinance.Ticker = MagicMock()
```

Since `MagicMock` is callable (it implements `__call__`), code that does `yf.Ticker("AAPL")` still works — it just returns another mock instead of a real `Ticker` instance.

## The `.return_value` Chain

Here's the full pattern used in the data-fetcher tests:

```python
MOCK_HISTORICAL_DF = pd.DataFrame(
    {"Open": [1.0, 2.0], "High": [1.5, 2.5], "Low": [0.5, 1.5], "Close": [1.2, 2.2], "Volume": [100, 200]}
)

def test_fetcher_fetch_historical_returns_dataframe(mocker, kwargs_fetcher):
    kwargs = copy.deepcopy(kwargs_fetcher)
    mock_ticker = mocker.patch("yfinance.Ticker")
    mock_ticker.return_value.history.return_value = MOCK_HISTORICAL_DF
    tested = data_fetcher.fetcher.Fetcher(**kwargs)
    df = tested.fetch_historical()
    assert isinstance(df, pd.DataFrame)
```

The chain `mock_ticker.return_value.history.return_value` mirrors the production code step by step:

| Mock chain | Production code | What happens |
|---|---|---|
| `mock_ticker(...)` → `.return_value` | `yf.Ticker("AAPL")` | Calling the mock class returns a mock instance |
| `.history` | `ticker.history` | Accessing an attribute on the mock instance |
| `.history(...)` → `.return_value` | `ticker.history(period=...)` | Calling that attribute returns `MOCK_HISTORICAL_DF` |

`MagicMock` auto-creates attributes on access. Each `.` in the chain creates a new nested mock automatically. You only assign a real value at the end of the chain — the point where you want to control the output.

## Why Mock the Whole Class?

An alternative is to mock only the method:

```python
mocker.patch("yfinance.Ticker.history", return_value=MOCK_HISTORICAL_DF)
```

This would work, but the test would still call the **real** `yf.Ticker("AAPL")` constructor. That constructor may reach Yahoo's servers or fail without network access.

By mocking the entire class, nothing from `yfinance` ever executes. The test is fully isolated from the network.

## Automatic Cleanup

`mocker.patch` restores the original object when the test ends — no manual teardown needed. Each test gets a fresh mock (or the real object) depending on whether it uses `mocker.patch`.
