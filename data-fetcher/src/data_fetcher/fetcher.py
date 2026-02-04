import yfinance as yf


class Fetcher:
    def __init__(self, symbol, period="7d", interval="1h", **kwargs):
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.ticker = yf.Ticker(symbol)

    def fetch_historical(self):
        return self.ticker.history(period=self.period, interval=self.interval)

    def fetch_latest(self):
        df = self.ticker.history(period="1d", interval="1m")
        return df.iloc[-1] if len(df) > 0 else None
