---
name: testing
description: Writing and running tests in the data-processing monorepo. Use when adding, modifying, or debugging tests, working with the YAML-driven parametrize-tests setup, conftest fixtures, or test config sweeps.
---

# Testing in data-processing

**Do not run pytest unless the user explicitly asks.** Writing and editing tests
is fine; executing them (`uv run pytest`, `pytest`, or via any tool) is not,
absent an explicit request. After changing tests, stop and report — offer to run
them, but wait for the go-ahead. (Build/install/run and lint commands live in the
`build-and-style` skill.)

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

## Mocking external / network calls

- Use the **`mocker`** fixture (pytest-mock), not `unittest.mock`: it auto-reverts
  at teardown, needs no decorator/`with` nesting, and reads top-to-bottom.
- Keep tests offline and deterministic — never hit yfinance live. Patch the
  `Ticker` class and feed a fixed DataFrame:

  ```python
  mock_ticker = mocker.patch("yfinance.Ticker")
  mock_ticker.return_value.history.return_value = MOCK_HISTORICAL_DF
  ```

  Two `return_value`s: one for calling `Ticker(...)`, one for calling `.history(...)`.
- Examples: `data-fetcher/tests/test_fetcher.py`,
  `stock-analyzer/stock-analyzer-activator/tests/test_fetch_to_bin.py`.
- `pytest-mock` must be in that module's `pyproject.toml` test deps for `mocker`
  to resolve when its tests run standalone.

## How the suite runs in pre-commit

- The `pytest` hook runs the **whole suite** but `--ignore`s untracked test files
  (they won't be pushed, so CI won't run them). Pre-commit's stash already hides
  unstaged edits to tracked files, so the run reflects the staged snapshot.
- CI skips that hook (`SKIP=pytest`) and runs a plain `uv run pytest` on a clean
  checkout — no untracked files exist there, so no `--ignore` is needed.

## Pitfalls

- Avoid duplicate test-file basenames across modules — they cause pytest
  collection collisions. Fix by giving the files unique names; do not add
  `__init__.py` files or reach for `importlib`.
