import copy
import pathlib

import stock_analyzer_activator.activator


def test_prep_kwargs_runs_fetch_to_bin(kwargs_activator, tmp_path):
    kwargs = copy.deepcopy(kwargs_activator)
    kwargs["input"]["path"] = str(tmp_path / "out.bin")
    stock_analyzer_activator.activator.Activator._prep_kwargs(kwargs)
    assert pathlib.Path(kwargs["input"]["path"]).exists()
