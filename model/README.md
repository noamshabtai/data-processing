# model

PyTorch LSTM neural network for time-series prediction.

## Components

### LSTM

Stacked LSTM layers followed by a fully connected output layer.

**Parameters:**
- `input_dim` - Number of input features
- `hidden_dim` - LSTM hidden state dimension
- `output_dim` - Number of output values
- `num_layers` - Number of stacked LSTM layers (default: 2)

### Predictor

Inference wrapper around the LSTM model.

## Usage

```python
from model.predictor import Predictor

predictor = Predictor(input_dim=6, hidden_dim=32, output_dim=1, num_layers=2)
predictor.load("model_weights.pt")
prediction = predictor.predict(features)  # numpy array in, numpy array out
```

## Dependencies

- torch
- numpy
