import copy

import data_fetcher.fetcher
import numpy as np
import stock_analyzer_activator.fetch_to_bin


def test_fetch_to_bin_writes_field_as_float32(kwargs_fetch_to_bin, tmp_path):
    kwargs = copy.deepcopy(kwargs_fetch_to_bin)
    kwargs["output_path"] = str(tmp_path / "out.bin")
    expected = (
        data_fetcher.fetcher.Fetcher(symbol=kwargs["symbol"], period=kwargs["period"], interval=kwargs["interval"])
        .fetch_historical()[kwargs["field"]]
        .to_numpy(dtype=np.float32)
    )
    stock_analyzer_activator.fetch_to_bin.fetch_to_bin(kwargs)
    written = np.fromfile(kwargs["output_path"], dtype=np.float32)
    np.testing.assert_array_equal(written, expected)
