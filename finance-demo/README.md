# finance-demo

Integrated demo pipeline that combines data fetching, feature extraction, and LSTM prediction into a continuous processing loop.

## Architecture

Extends the `System` and `Activator` base classes from the [signal-processing](https://github.com/noamshabtai/signal-processing) framework.

**Data flow:**
1. Fetch latest stock data
2. Buffer input data with configurable window
3. Extract features (trend + FFT) from the buffered window
4. Run LSTM prediction on extracted features
5. Log prediction and repeat

## Configuration

```yaml
input:
  symbol: AAPL
  period: 7d
  interval: 1h
system:
  input_buffer:
    step_size: 10
    buffer_size: 50
    channel_shape: [1]
    dtype: float32
  features:
    window_size: 20
  predictor:
    input_dim: 6
    hidden_dim: 32
    output_dim: 1
    num_layers: 2
poll_interval: 60
```

## Dependencies

- signal-processing (activator, system, buffer)
- data-fetcher
- feature-extraction
- model
