# parametrize-tests

YAML-based test parametrization utilities for pytest.

## Features

- Define test cases in YAML config files
- `sweep` sections generate cartesian product of parameter combinations
- `base` sections define defaults inherited by all cases
- Pytest fixtures are auto-generated from YAML configs

## YAML Config Format

```yaml
base:
  common_param: value

cases:
  - name: basic case
    param: 1
  - name: sweep case
    param:
      sweep: [a, b, c]
```

## Usage in conftest.py

```python
import parametrize_tests.fixtures

config_dir = tests_dir / "config"
module = sys.modules[__name__]

parametrize_tests.fixtures.setattr_kwargs("my_fixture", config_dir, module)
parametrize_tests.fixtures.setattr_project_dir(tests_dir, module)
```

This registers a `kwargs_my_fixture` pytest fixture parametrized from `config/my_fixture.yaml`.

## Dependencies

- deepmerge >= 2.0
- pyyaml >= 6.0.3
