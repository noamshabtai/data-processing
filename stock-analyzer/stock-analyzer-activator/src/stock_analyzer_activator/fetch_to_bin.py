import pathlib

import data_fetcher.fetcher
import numpy as np


def fetch_to_bin(**kwargs):
    fetcher = data_fetcher.fetcher.Fetcher(
        symbol=kwargs["symbol"], period=kwargs["period"], interval=kwargs["interval"]
    )
    values = fetcher.fetch_historical()[kwargs["field"]].to_numpy(dtype=np.float32)
    output_path = pathlib.Path(kwargs["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values.tofile(output_path)


if __name__ == "__main__":
    import sys

    import yaml

    with open(sys.argv[1]) as f:
        fetch_to_bin(**yaml.safe_load(f))
