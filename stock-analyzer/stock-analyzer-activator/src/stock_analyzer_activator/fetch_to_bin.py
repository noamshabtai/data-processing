import pathlib

import data_fetcher.fetcher
import numpy as np


def fetch_to_bin(config):
    fetcher = data_fetcher.fetcher.Fetcher(
        symbol=config["symbol"], period=config["period"], interval=config["interval"]
    )
    values = fetcher.fetch_historical()[config["field"]].to_numpy(dtype=np.float32)
    output_path = pathlib.Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values.tofile(output_path)


if __name__ == "__main__":
    import sys

    import yaml

    with open(sys.argv[1]) as f:
        fetch_to_bin(yaml.safe_load(f))
