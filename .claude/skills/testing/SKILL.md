---
name: testing
description: Writing and running tests in the data-processing monorepo. Use when adding, modifying, or debugging tests, working with the YAML-driven parametrize-tests setup, conftest fixtures, or test config sweeps.
---

# Testing in data-processing

Run the full suite with:

```
uv run pytest
```

## YAML-parametrized tests

- Tests are parametrized via YAML config files in each module's `tests/config/`
  directory.
- The `parametrize-tests` package (from the sibling `signal-processing` repo)
  parses YAML configs and generates pytest fixtures.
- YAML configs support:
  - `sweep` — cartesian product of parameter values.
  - `base` — inherited defaults.
- All modules follow the same `conftest.py` pattern using
  `parametrize_tests.fixtures`.

## conftest pattern

- `conftest.py` registers YAML-based parametrized fixtures via
  `parametrize_tests.fixtures.setattr_kwargs`.
- Deep copy of kwargs in tests ensures isolation between parametrized cases.

## Linting in tests

Ruff ignores in test files: `S101` (assert) and `PLR2004` (magic values). See the
`build-and-style` skill for the full ruleset.

## Pitfalls

- Avoid duplicate test-file basenames across modules — they cause pytest
  collection collisions. Fix by giving the files unique names; do not add
  `__init__.py` files or reach for `importlib`.
