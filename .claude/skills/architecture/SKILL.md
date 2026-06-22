---
name: architecture
description: Understanding the data-processing pipeline architecture — end-to-end data flow, the LSTM model, key classes and their file locations, and the module dependency graph. Use when navigating the codebase, locating a class, adding a pipeline stage, or reasoning about how the stock-analyzer wires components together.
---

# data-processing architecture

`data-processing` is a modular Python monorepo for financial time-series analysis
and prediction. It combines signal processing, feature extraction, and deep
learning (LSTM) to analyze stock data.

## Module dependency graph

```
stock-analyzer -> feature-extraction (numpy, scipy)
               -> model (torch, numpy)
               -> data-fetcher (yfinance, pandas)
               -> signal-processing (activator, system, buffer)
```

## Data flow pipeline

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

The `StockAnalyzer.connect()` method in `stock-analyzer` defines data routing
using `match`/`case`:
- buffer output → `FeatureExtraction.execute()`
- feature output → `Trainer.execute()`

## Model architecture

2-layer LSTM → single Linear output layer. No hidden fully-connected layers.

```
LSTM(input_dim=6, hidden_dim=32, num_layers=2) → Linear(32→1) → prediction
```

## Key classes & locations

| Class | Module | Path |
|---|---|---|
| `Fetcher` | data-fetcher | `data-fetcher/src/data_fetcher/fetcher.py` |
| `FeatureExtraction` | feature-extraction | `feature-extraction/src/feature_extraction/feature_extraction.py` |
| `Model` | model | `model/src/model/model.py` |
| `Trainer` | model | `model/src/model/trainer.py` |
| `StockAnalyzer` | stock-analyzer | `stock-analyzer/src/stock_analyzer/stock_analyzer.py` |

## Module layout patterns

- Each module uses a `src/` layout with the hatchling build backend.
- Test configs live in `tests/config/*.yaml` (see the `testing` skill).

## Educational resources

- [`docs/lstm-walkthrough.md`](../../../docs/lstm-walkthrough.md) — Deep dive into
  the project pipeline, LSTM internals, and PyTorch concepts.
