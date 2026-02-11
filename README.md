# data-processing

Modular Python framework for financial time-series analysis and prediction using signal processing, feature extraction, and deep learning.

## Architecture

```
Stock Data (yfinance) -> Input Buffer -> Feature Extraction -> LSTM Model -> Prediction
                                         ├── Trend (MA, slope)
                                         └── FFT (frequency analysis)
```

## Modules

| Module | Description |
|---|---|
| [data-fetcher](data-fetcher/) | Fetch historical and real-time stock data via Yahoo Finance |
| [feature-extraction](feature-extraction/) | Trend and FFT feature extractors for time-series data |
| [model](model/) | PyTorch LSTM neural network for prediction |
| [finance-demo](finance-demo/) | Integrated pipeline combining all modules |
| [stock-analyzer](stock-analyzer/) | CLI tool for training models and running predictions |
| [parametrize-tests](parametrize-tests/) | YAML-based pytest parametrization utility |

## Prerequisites

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- [signal-processing](https://github.com/noamshabtai/signal-processing) repository cloned as a sibling directory:

```bash
git clone https://github.com/noamshabtai/signal-processing.git ../signal-processing
```

## Setup

```bash
uv sync
```

## Testing

```bash
# Run full test suite
pytest -n 6

# Run pre-commit hooks
pre-commit run --all-files
```

## Code Style

- Black formatter (line length 120)
- Ruff linter
- Pre-commit hooks enforced on all commits
