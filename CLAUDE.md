# CLAUDE.md

## Project Overview

data-processing is a modular Python monorepo for financial time-series analysis and prediction. It combines signal processing, feature extraction, and deep learning (LSTM) to analyze stock data.

## Repository Structure

```
data-processing/
├── data-fetcher/          # Stock data acquisition via yfinance
├── feature-extraction/    # Trend and FFT feature extractors
├── model/                 # PyTorch LSTM neural network
├── finance-demo/          # Integrated demo pipeline
├── stock-analyzer/        # CLI tool for training and prediction
├── parametrize-tests/     # YAML-based pytest parametrization (shared with signal-processing)
└── .github/workflows/     # CI/CD
```

## External Dependencies

This project depends on the [signal-processing](https://github.com/noamshabtai/signal-processing) repository, which must be cloned as a sibling directory (`../signal-processing`). It provides `activator`, `system`, and `buffer` modules, plus the shared `parametrize-tests` package.

## Build & Run

- **Python:** >=3.12
- **Package manager:** uv
- **Install:** `uv sync`
- **Run tests:** `pytest -n 6`
- **Pre-commit:** `pre-commit run --all-files`

## Code Style

- **Formatter:** Black (line length 120)
- **Linter:** Ruff with rules PERF, PL, B, S, F, W, E, I, TID
  - Ignored in tests: S101 (assert), PLR2004 (magic values)
  - Ignored globally: PLR2004, PLW0603, PLW0602

## Testing

- Tests are parametrized via YAML config files in each module's `tests/config/` directory
- The `parametrize-tests` package parses YAML configs and generates pytest fixtures
- YAML configs support `sweep` (cartesian product) and `base` (inherited defaults) sections
- All modules follow the same conftest.py pattern using `parametrize_tests.fixtures`

## Module Dependency Graph

```
stock-analyzer -> finance-demo -> feature-extraction (numpy, scipy)
                               -> model (torch, numpy)
                               -> data-fetcher (yfinance, pandas)
                               -> signal-processing (activator, system, buffer)
```

## Data Flow Pipeline

```
AAPL price tick (yfinance Fetcher)
    │
    ▼
Input Buffer (sliding window, e.g. last 50 points)
    │
    ▼
FeatureExtraction.execute()
  ├── _extract_trend → [moving_avg, slope, slope_std]
  └── _extract_fft   → [dom_freq, dom_mag, mean_mag]
    │
    ▼
Feature vector: 6 values (this is why input_dim=6 in configs)
    │
    ▼
LSTM (2 layers, hidden_dim=32) → Linear(32→1) → prediction
```

The `System.connect()` method in `finance-demo` defines data routing using `match`/`case`:
- buffer output → `FeatureExtraction.execute()`
- feature output → `Model.execute()`

## Key Classes & Locations

| Class | Module | Path |
|---|---|---|
| `Fetcher` | data-fetcher | `data-fetcher/src/data_fetcher/fetcher.py` |
| `FeatureExtraction` | feature-extraction | `feature-extraction/src/feature_extraction/feature_extraction.py` |
| `Model` | model | `model/src/model/model.py` |
| `System` | finance-demo | `finance-demo/src/finance_demo/system.py` |

## Model Architecture

2-layer LSTM → single Linear output layer. No hidden fully-connected layers.

```
LSTM(input_dim=6, hidden_dim=32, num_layers=2) → Linear(32→1) → prediction
```

## CLI Usage

```bash
stock-analyzer train --symbol AAPL --period 1y --epochs 10 --output model.pt
stock-analyzer predict --symbol AAPL --model model.pt
```

## Key Patterns

- Each module uses `src/` layout with hatchling build backend
- Test configs live in `tests/config/*.yaml`
- `conftest.py` registers YAML-based parametrized fixtures via `parametrize_tests.fixtures.setattr_kwargs`
- Deep copy of kwargs in tests ensures isolation between parametrized cases

## Educational Resources

- [`docs/lstm-walkthrough.md`](docs/lstm-walkthrough.md) — Deep dive into the project pipeline, LSTM internals, and PyTorch concepts
