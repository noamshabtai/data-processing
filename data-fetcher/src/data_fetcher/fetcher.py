import yfinance as yf


class Fetcher:
    def __init__(self, **kwargs):
        self.symbol = kwargs.get("symbol", "GOOGL")
        self.period = kwargs.get("period", "7d")
        self.interval = kwargs.get("interval", "1h")
        self.ticker = yf.Ticker(self.symbol)

    def fetch_historical(self):
        return self.ticker.history(period=self.period, interval=self.interval)

    def fetch_latest(self):
        df = self.ticker.history(period="1d", interval="1m")
        return df.iloc[-1] if len(df) > 0 else None
