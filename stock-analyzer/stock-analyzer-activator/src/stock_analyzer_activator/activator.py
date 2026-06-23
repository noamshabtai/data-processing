import activator.offline
import stock_analyzer.predictor

import stock_analyzer_activator.fetch_to_bin


class Activator(activator.offline.Activator):
    def __init__(self, **kwargs):
        fetch_cfg = kwargs["input"].get("fetch_to_bin")
        self.symbol = fetch_cfg["symbol"] if fetch_cfg else None
        if fetch_cfg:
            fetch_kwargs = {**fetch_cfg, "output_path": kwargs["input"]["path"]}
            stock_analyzer_activator.fetch_to_bin.fetch_to_bin(**fetch_kwargs)
        super().__init__(stock_analyzer.predictor.System, **kwargs["system"])

    def post_figure_hook(self, plt, module, data):
        plt.title(f"{self.symbol} — {module}")
        if module == "trainer":
            plt.ylabel("price ($)")
