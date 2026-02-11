# data-fetcher

Stock data acquisition module using the Yahoo Finance API.

## Usage

```python
from data_fetcher.fetcher import Fetcher

fetcher = Fetcher(symbol="AAPL", period="7d", interval="1h")

# Get historical OHLCV data as a DataFrame
df = fetcher.fetch_historical()

# Get the most recent data point as a Series
latest = fetcher.fetch_latest()
```

## API

### `Fetcher(symbol, period="7d", interval="1h")`

- `symbol` - Stock ticker (e.g. `"AAPL"`)
- `period` - Time range for historical data (e.g. `"7d"`, `"1y"`)
- `interval` - Data granularity (e.g. `"1h"`, `"1d"`)

**Methods:**
- `fetch_historical()` - Returns a pandas DataFrame with Open, High, Low, Close, Volume columns
- `fetch_latest()` - Returns the most recent 1-minute data point as a pandas Series

## Dependencies

- yfinance
- pandas
