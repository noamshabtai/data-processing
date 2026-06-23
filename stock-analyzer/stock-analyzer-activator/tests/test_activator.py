import copy

import stock_analyzer.predictor
import stock_analyzer_activator.activator


def test_activator_fetches_to_bin(mocker, kwargs_activator):
    kwargs = copy.deepcopy(kwargs_activator)
    mock_base_init = mocker.patch("activator.offline.Activator.__init__", return_value=None)
    mock_fetch = mocker.patch("stock_analyzer_activator.fetch_to_bin.fetch_to_bin")

    activator = stock_analyzer_activator.activator.Activator(**kwargs)

    assert activator.symbol == kwargs["input"]["fetch_to_bin"]["symbol"]
    mock_fetch.assert_called_once_with(
        **kwargs["input"]["fetch_to_bin"],
        output_path=kwargs["input"]["path"],
    )
    mock_base_init.assert_called_once_with(stock_analyzer.predictor.System, **kwargs["system"])


def test_activator_skips_fetch_when_unconfigured(mocker, kwargs_activator):
    kwargs = copy.deepcopy(kwargs_activator)
    del kwargs["input"]["fetch_to_bin"]
    mocker.patch("activator.offline.Activator.__init__", return_value=None)
    mock_fetch = mocker.patch("stock_analyzer_activator.fetch_to_bin.fetch_to_bin")

    activator = stock_analyzer_activator.activator.Activator(**kwargs)

    mock_fetch.assert_not_called()
    assert activator.symbol is None
