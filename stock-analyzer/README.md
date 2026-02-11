# stock-analyzer

Command-line interface for stock analysis operations.

## Installation

Installed as part of the data-processing monorepo via `uv sync`. Provides the `stock-analyzer` command.

## Commands

### train

Train a model on historical stock data.

```bash
stock-analyzer train --symbol AAPL --period 1y --epochs 10 --output model.pt
```

| Flag | Description | Default |
|---|---|---|
| `--symbol` | Stock ticker (required) | - |
| `--period` | Training data timespan | `1y` |
| `--epochs` | Training iterations | `10` |
| `--output` | Model save path (required) | - |

### predict

Run predictions using a trained model.

```bash
stock-analyzer predict --symbol AAPL --model model.pt [--simulate]
```

| Flag | Description | Default |
|---|---|---|
| `--symbol` | Stock ticker (required) | - |
| `--model` | Path to trained model (required) | - |
| `--simulate` | Run in simulation mode | `false` |

## Dependencies

- finance-demo
