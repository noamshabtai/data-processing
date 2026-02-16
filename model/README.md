# model

PyTorch LSTM neural network for time-series prediction.

## Model

Single class owning the full lifecycle: init, inference, training, save, and load. Internally uses a 2-layer LSTM followed by a fully connected output layer.

**Parameters:**
- `input_dim` - Number of input features
- `hidden_dim` - LSTM hidden state dimension
- `output_dim` - Number of output values (default: 1)
- `num_layers` - Number of stacked LSTM layers (default: 2)

## Usage

```python
from model.model import Model

m = Model(input_dim=6, hidden_dim=32, output_dim=1, num_layers=2)

# Inference: numpy in, numpy out
prediction = m.execute(features)

# Training
loss = m.backward(data, targets, epochs=10, lr=0.001)

# Save and load weights
m.save("model_weights.pt")
m.load("model_weights.pt")
```

## Dependencies

- torch
- numpy
