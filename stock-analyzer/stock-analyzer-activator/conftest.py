import pathlib
import sys

import parametrize_tests.kwargs

tests_dir = pathlib.Path(__file__).parent / "tests"
config_dir = tests_dir / "config"
module = sys.modules[__name__]
for fixture in [
    "activator",
    "fetch_to_bin",
]:
    parametrize_tests.kwargs.setattr_kwargs(fixture, config_dir, module)
