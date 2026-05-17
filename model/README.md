# model

PyTorch LSTM for time-series prediction. Split into two classes:

- `Model` — the `torch.nn.Module` (LSTM + Linear). Pure inference, plus weight loading for swapping in pretrained weights.
- `Trainer` — owns the training lifecycle: optimizer, `backward` loop, numpy-friendly `execute`, and full checkpointing (model + optimizer state).

## Model

**Parameters:**
- `input_dim` - Number of input features
- `hidden_dim` - LSTM hidden state dimension
- `output_dim` - Number of output values (default: 1)
- `num_layers` - Number of stacked LSTM layers (default: 2)

```python
from model.model import Model

m = Model(input_dim=6, hidden_dim=32, output_dim=1, num_layers=2)

# Load pretrained weights for inference
m.load_weights("weights.pt")
```

## Trainer

Wraps a `Model` and an `Adam` optimizer. The optimizer is persistent on the instance so its adaptive state (momentum, second moments, step count) survives across `backward` calls and is captured by `save_checkpoint`.

**Parameters:**
- `network` - dict of `Model` kwargs
- `epochs`, `learning_rate`, `batch_size`, `data_shuffle`

```python
from model.trainer import Trainer

t = Trainer(network={"input_dim": 6, "hidden_dim": 32}, epochs=10, learning_rate=0.001, batch_size=4)

# Inference: numpy in, numpy out
prediction = t.execute(features)

# Training
epoch_losses = t.backward(data, targets)

# Checkpointing: model weights + optimizer state in one file
t.save_checkpoint("checkpoint.pt")
t.load_checkpoint("checkpoint.pt")
```

## Dependencies

- torch
- numpy
