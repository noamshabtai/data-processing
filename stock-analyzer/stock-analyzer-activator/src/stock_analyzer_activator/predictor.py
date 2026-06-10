import activator.offline
import stock_analyzer.predictor

import stock_analyzer_activator.fetch_to_bin


class Activator(activator.offline.Activator):
    def __init__(self, **kwargs):
        self.symbol = kwargs["symbol"]
        self._prep_kwargs(kwargs)
        super().__init__(stock_analyzer.predictor.System, **kwargs)

    @classmethod
    def _prep_kwargs(cls, kwargs):
        if "fetch_to_bin" not in kwargs["input"]:
            return
        fetch_cfg = {
            **kwargs["input"]["fetch_to_bin"],
            "symbol": kwargs["symbol"],
            "output_path": kwargs["input"]["path"],
        }
        stock_analyzer_activator.fetch_to_bin.fetch_to_bin(fetch_cfg)

    def post_figure_hook(self, plt, module, data):
        plt.title(f"{self.symbol} — {module}")
        if module == "trainer":
            plt.ylabel("price ($)")
