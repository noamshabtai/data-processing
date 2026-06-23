---
name: build-and-style
description: Build, install, run, and code-style conventions for the data-processing monorepo, including the required sibling signal-processing dependency. Use when setting up the environment, installing dependencies, running the project, or applying formatting and linting rules.
---

# Build & style — data-processing

## Build & run

- **Python:** >=3.12
- **Package manager:** uv
- **Install:** `uv sync`
- **Run tests:** `uv run pytest`
- **Pre-commit:** `pre-commit run --all-files`. For how the `pytest` hook and CI
  decide which tests run, see the `testing` skill.

## External dependency

This project depends on the
[signal-processing](https://github.com/noamshabtai/signal-processing)
repository, which must be cloned as a **sibling directory** (`../signal-processing`).
It provides the `activator`, `system`, and `buffer` modules, plus the shared
`parametrize-tests` package.

## Code style

- **Formatter:** Black (line length 120)
- **Linter:** Ruff with rules `PERF, PL, B, S, F, W, E, I, TID`
  - Ignored in tests: `S101` (assert), `PLR2004` (magic values)
  - Ignored globally: `PLR2004`, `PLW0603`, `PLW0602`
