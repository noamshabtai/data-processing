# LSTM & Project Pipeline Walkthrough

An educational guide to how the data-processing project works, from raw stock prices to LSTM predictions. Covers the full pipeline, PyTorch fundamentals, LSTM internals, and neural network concepts.

## Table of Contents

- [The Pipeline](#the-pipeline)
  - [Step 1: Fetching Stock Data](#step-1-fetching-stock-data-data-fetcher)
  - [Step 2: Buffering](#step-2-buffering-signal-processing--external)
  - [Step 3: Feature Extraction](#step-3-feature-extraction-feature-extraction)
  - [Step 4: LSTM Prediction](#step-4-lstm-prediction-model)
  - [Step 5: System Integration](#step-5-system-integration-finance-demo)
  - [Step 6: CLI](#step-6-cli-stock-analyzer)
  - [Full Data Flow Diagram](#full-data-flow-diagram)
- [PyTorch Concepts](#pytorch-concepts)
  - [nn.Module](#nnmodule)
  - [nn.LSTM](#nnlstm)
  - [nn.Linear](#nnlinear)
  - [Inference with torch.no_grad](#inference-with-torchno_grad)
- [Neural Network Concepts](#neural-network-concepts)
  - [nn.Linear vs Activations](#nnlinear-vs-activations)
  - [Hidden Layers](#hidden-layers)
  - [MLP vs LSTM](#mlp-vs-lstm)
- [Deep Dive: Inside the LSTM Cell](#deep-dive-inside-the-lstm-cell)
  - [The 4 Gates](#the-4-gates)
  - [Parameter Count](#parameter-count)
  - [Stacked Layers](#stacked-layers)
- [Deep Dive: Element-wise Operations and Data Flow](#deep-dive-element-wise-operations-and-data-flow)
  - [Element-wise Multiplication](#element-wise-multiplication)
  - [Cell State Flow](#cell-state-flow)
  - [Why "Long-Term" Memory?](#why-long-term-memory)
  - [Hidden State](#hidden-state)
  - [Input Concatenation](#input-concatenation)
  - [Layer 2 Inputs](#layer-2-inputs)
  - [Full Picture for One Timestep](#full-picture-for-one-timestep)
- [Deep Dive: Sigmoid vs Tanh](#deep-dive-sigmoid-vs-tanh)
- [Deep Dive: Training vs Streaming](#deep-dive-training-vs-streaming)
  - [Training (Offline)](#training-offline)
  - [Live Prediction (Streaming)](#live-prediction-streaming)
  - [batch_first=True](#batch_firsttrue)
  - [self.lstm() Return Values](#selflstm-return-values)
- [History](#history)

---

## The Pipeline

This project takes **live stock prices**, extracts **meaningful patterns** from them, and feeds those patterns into an **LSTM neural network** to predict future values:

```
Raw stock prices -> Signal features -> Neural network -> Prediction
```

### Step 1: Fetching Stock Data (`data-fetcher`)

Everything starts with real market data. `Fetcher` wraps the Yahoo Finance API:

```python
# data-fetcher/src/data_fetcher/fetcher.py
fetcher = Fetcher(symbol="AAPL", period="7d", interval="1h")
df = fetcher.fetch_historical()  # DataFrame with Open, High, Low, Close, Volume
```

This gives you a table of price candles — each row is one hour of AAPL trading. The key column you care about downstream is **Close** (the closing price at each interval).

`fetch_latest()` grabs the most recent 1-minute candle for live use.

### Step 2: Buffering (`signal-processing` — external)

Raw prices arrive one at a time. But to detect trends or frequencies, you need a **window** of recent data — say the last 50 points.

The `buffer` module (from the sibling signal-processing repo) accumulates incoming data points and provides a sliding window. The config controls this:

```yaml
input_buffer:
  step_size: 10     # process every 10 new samples
  buffer_size: 50   # keep a window of 50 points
  channel_shape: [1] # single channel (just price)
```

So you're always working with the last 50 price points.

### Step 3: Feature Extraction (`feature-extraction`)

Raw prices aren't great direct input for a neural network. Instead, you extract **features** — compact numbers that describe *what the signal looks like*.

The `FeatureExtraction` class extracts both trend and frequency features in a single `execute()` call:

```python
# feature-extraction/src/feature_extraction/feature_extraction.py
def execute(self, data):
    features = [self._extract_trend(data), self._extract_fft(data)]
    return np.concatenate(features)
```

#### `_extract_trend` — "Where is the price going?"

```python
def _extract_trend(self, data):
    moving_avg = np.convolve(data, np.ones(self.window_size) / self.window_size, mode="valid")
    slope = np.gradient(moving_avg)
    return np.array([moving_avg[-1], slope[-1], np.std(slope)])
```

Three values:
1. **`moving_avg[-1]`** — the current smoothed price (filters out noise)
2. **`slope[-1]`** — is the price going up or down *right now*?
3. **`np.std(slope)`** — how volatile is the trend? (stable vs. erratic)

#### `_extract_fft` — "Are there repeating patterns?"

```python
def _extract_fft(self, data):
    fft_result = np.fft.rfft(data)
    magnitudes = np.abs(fft_result)
    dominant_freq_idx = np.argmax(magnitudes[1:]) + 1
    return np.array([dominant_freq_idx, magnitudes[dominant_freq_idx], np.mean(magnitudes)])
```

FFT decomposes a signal into frequencies (like a prism splitting light into colors). Three values:
1. **`dominant_freq_idx`** — which frequency is strongest (is there a cycle?)
2. **`magnitudes[dominant_freq_idx]`** — how strong is that cycle?
3. **`np.mean(magnitudes)`** — overall signal energy

```python
# FeatureExtraction.execute() output: [ma, slope, slope_std, freq_idx, freq_mag, mean_mag]
```

You now have a **6-element feature vector** for each window of data.

### Step 4: LSTM Prediction (`model`)

This is the PyTorch part. The `Model` class owns the full lifecycle — init, inference, training, save, and load. Internally it uses an LSTM `nn.Module`:

```python
# model/src/model/model.py (internal nn.Module)
class _Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)          # process the sequence
        return self.fc(lstm_out[:, -1, :])   # take the last timestep
```

The `Model` class wraps this for inference and training:

```python
# model/src/model/model.py
def execute(self, features):
    self.model.eval()
    with torch.no_grad():                        # no gradient tracking (inference only)
        x = torch.from_numpy(features).unsqueeze(0)  # numpy -> tensor, add batch dim
        output = self.model(x)
        return output.numpy().squeeze()          # tensor -> numpy, remove batch dim
```

### Step 5: System Integration (`finance-demo`)

The `System` class wires the modules:

```python
# finance-demo/src/finance_demo/system.py
class System(system.system.System):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modules["features"] = FeatureExtraction(**kwargs.get("features", {}))
        self.modules["predictor"] = Model(**kwargs.get("predictor", {}))

    def connect(self, module):
        match module:
            case "features":
                self.inputs[module] = dict(data=self.input_buffer.get_window())
            case "predictor":
                self.inputs[module] = dict(features=self.outputs["features"])
```

The `connect` method defines the **data routing**: buffer output feeds into features, features output feeds into predictor.

The `Activator` runs the loop:

```python
# finance-demo/src/finance_demo/demo.py
def execute(self):
    self.running = True
    while self.running:
        frame = self._fetch_latest()       # get new price
        if frame is not None:
            prediction = self.process_frame(frame)  # buffer -> features -> LSTM
            if prediction is not None:
                self._log_prediction(frame, prediction)
        time.sleep(self.poll_interval)     # wait 60s, repeat
```

Every 60 seconds: fetch price, push through the pipeline, print prediction.

### Step 6: CLI (`stock-analyzer`)

The entry point for users:

```bash
stock-analyzer train --symbol AAPL --period 1y --epochs 10 --output model.pt
stock-analyzer predict --symbol AAPL --model model.pt
```

This is the thin command-line layer that orchestrates training and live prediction using all the modules above.

### Full Data Flow Diagram

```
AAPL price tick (yfinance)
    |
    v
Input Buffer (keeps last 50 points)
    |
    v
+------------------------+
|  FeatureExtraction     |
|    _extract_trend      |--> [moving_avg, slope, slope_std]
|    _extract_fft        |--> [dom_freq, dom_mag, mean_mag]
+------------------------+
    |
    v
Feature vector: [6 values]
    |
    v
LSTM (2 layers, hidden_dim=32)
    |
    v
Prediction (1 value)
```

### Testing

Every module has YAML-driven parametrized tests. Instead of hardcoding test params in Python, they're defined in `tests/config/*.yaml` and the `parametrize-tests` utility generates pytest fixtures from them. This makes it easy to add test cases without touching Python code.

---

## PyTorch Concepts

### nn.Module

Base class for all neural network layers in PyTorch. You define layers in `__init__` and the computation in `forward`.

### nn.LSTM

`nn.LSTM(input_dim=6, hidden_dim=32, num_layers=2, batch_first=True)` — a 2-layer LSTM. Each timestep receives 6 features. Internally it maintains a 32-dimensional hidden state that carries memory across timesteps. `batch_first=True` means the input shape is `(batch, sequence_length, 6)`.

### nn.Linear

`nn.Linear(32, 1)` — a fully connected layer that maps the 32-dim hidden state to 1 output value (the prediction).

`lstm_out[:, -1, :]` — from the output of all timesteps, take only the **last** one. The idea: the final hidden state has "seen" the entire sequence and encodes the most up-to-date understanding.

### Inference with torch.no_grad

- **`torch.no_grad()`** — tells PyTorch "don't track gradients." During training you need gradients for backpropagation, but during inference it's wasted memory/computation.
- **`unsqueeze(0)`** — adds a batch dimension. The model expects `(batch, seq, features)` but you're feeding one sample, so you go from shape `(seq, 6)` to `(1, seq, 6)`.
- **`squeeze()`** — removes that batch dimension from the output.

---

## Neural Network Concepts

### nn.Linear vs Activations

`nn.Linear` is NOT an activation. It's a **fully connected (dense) layer** — just a matrix multiplication plus a bias:

```
output = input @ W + b
```

Where `W` is a weight matrix and `b` is a bias vector. Purely linear math, no activation function.

An **activation** would be something like `nn.ReLU()`, `nn.Sigmoid()`, or `nn.Tanh()` — the nonlinear function applied *after* a linear layer.

### Hidden Layers

This model has **zero hidden fully-connected layers**. The LSTM does the heavy lifting (2 stacked LSTM layers internally), and then `nn.Linear(32, 1)` is just the **output layer**.

```
LSTM (2 recurrent layers, 32-dim state) -> Linear(32 -> 1) -> done
```

For comparison, here's what 2 hidden FC layers would look like:

```python
# CURRENT — no hidden FC layers (model/src/model/model.py, _Network class)
class _Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)       # 32 -> 1 (output)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
```

```python
# WITH 2 HIDDEN LAYERS
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)              # 32 -> 64 (hidden 1)
        self.relu1 = nn.ReLU()                             # activation
        self.fc2 = nn.Linear(64, 32)                       # 64 -> 32 (hidden 2)
        self.relu2 = nn.ReLU()                             # activation
        self.fc3 = nn.Linear(32, output_dim)               # 32 -> 1  (output)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]    # take last timestep
        x = self.relu1(self.fc1(x))  # hidden layer 1 + activation
        x = self.relu2(self.fc2(x))  # hidden layer 2 + activation
        output = self.fc3(x)          # output layer (no activation)
        return output
```

The pattern for each hidden layer is always:

```
Linear -> Activation -> Linear -> Activation -> ... -> Linear (output)
```

**Why activations between layers?** Without them, stacking linear layers is pointless — `Linear(Linear(x))` is still just a linear function. The activation (ReLU, etc.) adds nonlinearity, which is what lets the network learn complex patterns.

**Why no activation on the output layer?** Because this is a regression task (predicting a price). You want the output to be any real number, not squashed through ReLU (which clips negatives) or sigmoid (which squashes to 0-1).

Visual comparison:

```
CURRENT:
  LSTM[2 layers] -> Linear(32->1) -> prediction

WITH 2 HIDDEN:
  LSTM[2 layers] -> Linear(32->64) -> ReLU -> Linear(64->32) -> ReLU -> Linear(32->1) -> prediction
```

The current design keeps it simple — the LSTM layers themselves already have a lot of capacity, so a single output projection is often enough.

### MLP vs LSTM

#### MLP (Multi-Layer Perceptron)

What you might picture with 2 hidden layers — that's an MLP:

```
input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> output
```

An MLP sees all its input **at once**, as a flat vector. It has no concept of order or time. If you fed it 50 stock prices, it treats them as 50 independent numbers — it doesn't know that price #1 came before price #2.

#### LSTM (Long Short-Term Memory)

An LSTM processes data **one timestep at a time**, carrying a memory (hidden state) forward:

```
price_1  -> [LSTM cell] -> hidden_1
                             |
price_2  -> [LSTM cell] -> hidden_2
                             |
price_3  -> [LSTM cell] -> hidden_3
                             |
                        ... and so on
                             |
price_50 -> [LSTM cell] -> hidden_50 -> Linear -> prediction
```

At each step, the LSTM cell decides:
- **What to forget** from its memory (old trends that faded)
- **What to remember** from the new input (a sudden price spike)
- **What to output** as the current hidden state

#### Side by side

| | MLP | LSTM |
|---|---|---|
| Sees input as | Flat vector (all at once) | Ordered sequence (step by step) |
| Has memory | No | Yes — hidden state carries across steps |
| Good for | Tabular data, classification | Time series, sequences, text |
| Knows order | No — shuffling inputs changes nothing | Yes — order is everything |

#### What this project actually does

```
          LSTM part                    MLP part (sort of)
   +---------------------+         +--------------+
   | step 1 -> cell -> h1 |         |              |
   | step 2 -> cell -> h2 |         |              |
   | ...                  |  -->    | Linear(32->1) |  -->  prediction
   | step N -> cell -> hN |  last   |              |
   |   (2 stacked layers) |  step   |              |
   +---------------------+         +--------------+
```

The LSTM processes the sequence and builds up understanding over time. Then the final hidden state (`h_N`) gets passed through a single `Linear` layer to produce the prediction. That single `Linear` at the end is technically a 0-hidden-layer MLP — just an output projection.

So this model is: **LSTM for sequence processing** + **a tiny MLP head for output**. Not one or the other — they work together.

---

## Deep Dive: Inside the LSTM Cell

### The 4 Gates

An LSTM cell has **4 gates** — small neural networks that control information flow. At each timestep, the cell receives the current input `x_t` and the previous hidden state `h_{t-1}`, and maintains a **cell state** `C` (the long-term memory).

```
                    +-------------------------------------+
                    |           LSTM Cell                  |
                    |                                     |
  C_{t-1} -------->|---> [forget gate] ---> throw away    |--------> C_t
  (old memory)     |                      old memory      |   (new memory)
                   |---> [input gate]  ---> what to add    |
                   |---> [candidate]   ---> new values     |
                   |---> [output gate] ---> what to expose |
  h_{t-1} -------->|                                     |--------> h_t
  (old hidden)     |                                     |   (new hidden)
                   |                                     |
  x_t ------------>|                                     |
  (current input)  +-------------------------------------+
```

The 4 gates, step by step:

**1. Forget gate** — "what should I erase from memory?"

```
f = sigmoid(W_f . [h_{t-1}, x_t] + b_f)
```

Outputs values between 0 and 1 for each element of the cell state. 0 = completely forget, 1 = fully keep. Example: if the trend reversed, forget the old trend direction.

**2. Input gate** — "what new information is worth storing?"

```
i = sigmoid(W_i . [h_{t-1}, x_t] + b_i)
```

Also 0-1 values. Decides *which* parts of the memory to update.

**3. Candidate values** — "what are the new values to potentially store?"

```
C_new = tanh(W_c . [h_{t-1}, x_t] + b_c)
```

These are the proposed new values (between -1 and 1).

**4. Update the cell state** — apply forget and input:

```
C_t = f * C_{t-1} + i * C_new
      ^^^^^^^^^^^   ^^^^^^^^^^
      keep this     add this
```

This is the key insight of LSTMs. The cell state flows through time with only **element-wise** operations — no matrix multiplications. This is why LSTMs can remember things over long sequences (solves the vanishing gradient problem that plain RNNs have).

**5. Output gate** — "what part of my memory should I expose?"

```
o = sigmoid(W_o . [h_{t-1}, x_t] + b_o)
h_t = o * tanh(C_t)
```

The cell may know a lot, but only exposes the relevant part as its hidden state `h_t`.

### Parameter Count

Each gate is:

```
Linear(input_dim + hidden_dim -> hidden_dim) -> sigmoid (or tanh)
```

That's why LSTMs have so many parameters. With `input_dim=6` and `hidden_dim=32`, a single LSTM layer has 4 gates, each with a weight matrix of size `(6+32, 32)`. That's `4 x (38 x 32 + 32 bias) = 4 x 1248 = 4992` parameters in one layer.

### Stacked Layers

```python
self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
```

`num_layers=2` means two LSTM layers **stacked vertically**. The output sequence of layer 1 becomes the input sequence of layer 2:

```
                    Layer 1                    Layer 2
                  (input -> 32)               (32 -> 32)

timestep 1:   x_1 -> [LSTM cell] -> h1_1  ->  [LSTM cell] -> h2_1
                          |                        |
timestep 2:   x_2 -> [LSTM cell] -> h1_2  ->  [LSTM cell] -> h2_2
                          |                        |
timestep 3:   x_3 -> [LSTM cell] -> h1_3  ->  [LSTM cell] -> h2_3
                          |                        |
              ...        ...                      ...
                          |                        |
timestep N:   x_N -> [LSTM cell] -> h1_N  ->  [LSTM cell] -> h2_N
                                                                |
                                                          Linear(32->1)
                                                                |
                                                           prediction
```

- **Layer 1** sees the raw features (6 values per timestep) and produces a 32-dim hidden state at each step
- **Layer 2** sees Layer 1's hidden states (32 values per timestep) and produces its own 32-dim hidden states
- Only the **last hidden state** of Layer 2 (`h2_N`) goes to the output Linear layer

#### Are they "hidden" layers?

Yes, in the sense that they're between input and output and you never see their values directly. But they're not the same as MLP hidden layers:

| | MLP hidden layer | LSTM stacked layer |
|---|---|---|
| What it is | `Linear + activation` | Full LSTM cell (4 gates, cell state, hidden state) |
| Parameters | One weight matrix | Four weight matrices + biases |
| Has memory | No | Yes — carries state across timesteps |
| Sees data | Once | Once per timestep, sequentially |

#### Why stack them?

Same intuition as MLP — more layers = more abstraction:

- **Layer 1** might learn low-level patterns: "price is rising", "there's a dip"
- **Layer 2** takes those patterns and learns higher-level ones: "this looks like a reversal", "this dip is within a larger uptrend"

In this project's code, `num_layers=2` is set via the YAML config:

```yaml
model:
  input_dim: 6       # 6 features from the extraction pipeline
  hidden_dim: 32     # each LSTM layer has 32-dim hidden state
  output_dim: 1      # predict one value
  num_layers: 2      # two stacked LSTM layers
```

---

## Deep Dive: Element-wise Operations and Data Flow

### Element-wise Multiplication

`*` here means **element-wise multiplication**, not matrix multiplication. Every value in `f` multiplies the corresponding value in `C_{t-1}`, one by one.

`f` is a vector of 32 values, each between 0 and 1 (because sigmoid). `C_{t-1}` is also a vector of 32 values. So:

```
f         = [0.9,  0.1,  0.7,  ...]    (32 values, each 0-1)
C_{t-1}   = [3.2, -1.5,  0.8,  ...]    (32 values, the memory)

f * C_{t-1} = [2.88, -0.15, 0.56, ...]  (keep 90% of slot 1, erase 90% of slot 2, etc.)
```

Same with `i * C_new` — the input gate `i` controls how much of each proposed new value actually gets written to memory.

### Cell State Flow

`C_t` is used in the very next line to produce the hidden state:

```
C_t = f * C_{t-1} + i * C_new       <-- update memory
h_t = o * tanh(C_t)                 <-- read from memory
```

And then `C_t` gets passed to the **next timestep** as `C_{t-1}`. It's a chain:

```
C_0 -> C_1 -> C_2 -> C_3 -> ... -> C_N
```

The cell state is the **internal memory** that flows across time. The hidden state `h_t` is a filtered view of it — what the cell chooses to expose.

### Why "Long-Term" Memory?

The cell state `c` flows through time with only element-wise operations (`*` and `+`). No matrix multiplications, no squashing functions applied directly to it. Gradients flow back through it without shrinking, so it can carry information across many timesteps.

In a plain RNN, everything passes through matrix multiplications at every step, so signals decay quickly (the vanishing gradient problem). The cell state is a protected highway that avoids that.

The cell state itself is **unbounded** — it accumulates values over time through `c_t = f * c_{t-1} + i * candidate`. The forget gate can scale it down and the input gate adds new values, but there's no clamp on the result. The hidden state `h` is bounded though, because `h_t = o * tanh(c_t)` squashes the cell state to -1 to 1 before the output gate filters it.

So `c` is a raw accumulator, `h` is a bounded view of it.

### Hidden State

`h_t` is a single vector of 32 values. It's the output of the LSTM cell at timestep `t`. Not "all outputs stacked" — just one timestep's output.

Over a whole sequence, you get a **series** of hidden states:

```
h_1, h_2, h_3, ..., h_N     (each is 32 values)
```

PyTorch returns all of them stacked as `lstm_out` with shape `(batch, N, 32)`. Then the code takes only the last one:

```python
lstm_out[:, -1, :]   # shape: (batch, 32) — just h_N
```

### Input Concatenation

Inside the LSTM cell, the notation `[h_{t-1}, x_t]` means **concatenation**. If `h` is 32 values and `x` is 6 values, you concatenate them into one 38-value vector:

```
h_{t-1} = [h1, h2, ..., h32]         (32 values)
x_t     = [x1, x2, ..., x6]          (6 values)

[h_{t-1}, x_t] = [h1, h2, ..., h32, x1, ..., x6]   (38 values)
```

Then each gate multiplies this 38-value vector by a weight matrix:

```
f = sigmoid(W_f . [38 values] + b_f)
              ^^^^
         W_f is a (38 x 32) matrix
```

So within **Layer 1**, each gate is essentially a `Linear(38 -> 32) + sigmoid`.

### Layer 2 Inputs

Layer 2 does **not** see the original 6 input features. It only sees Layer 1's hidden states.

Layer 2 concatenates **its own** previous hidden state with Layer 1's output:

```
Layer 1, timestep t:  [h1_{t-1}(32), x_t(6)]     -> concat -> 38 values -> gates -> h1_t (32)
                                                                                       |
Layer 2, timestep t:  [h2_{t-1}(32), h1_t(32)]    -> concat -> 64 values -> gates -> h2_t (32)
```

So the weight matrices are different sizes per layer:

```
Layer 1 gates: W is (6 + 32)  x 32 = 38 x 32   (input + own hidden)
Layer 2 gates: W is (32 + 32) x 32 = 64 x 32   (layer 1 output + own hidden)
```

### Full Picture for One Timestep

```
x_t (6 values)
  |
  v
+--------------------------------------------------+
| LSTM Layer 1                                     |
|                                                  |
|  concat [h1_{t-1}, x_t] -> 38 values            |
|    +- forget gate: Linear(38->32) + sigmoid -> f |
|    +- input gate:  Linear(38->32) + sigmoid -> i |
|    +- candidate:   Linear(38->32) + tanh -> C_n  |
|    +- output gate: Linear(38->32) + sigmoid -> o |
|                                                  |
|  C1_t = f * C1_{t-1} + i * C_n                  |
|  h1_t = o * tanh(C1_t)                          |
+----------------------+---------------------------+
                       | h1_t (32 values)
                       v
+--------------------------------------------------+
| LSTM Layer 2                                     |
|                                                  |
|  concat [h2_{t-1}, h1_t] -> 64 values           |
|    +- forget gate: Linear(64->32) + sigmoid -> f |
|    +- input gate:  Linear(64->32) + sigmoid -> i |
|    +- candidate:   Linear(64->32) + tanh -> C_n  |
|    +- output gate: Linear(64->32) + sigmoid -> o |
|                                                  |
|  C2_t = f * C2_{t-1} + i * C_n                  |
|  h2_t = o * tanh(C2_t)                          |
+----------------------+---------------------------+
                       | h2_t (32 values)
                       v
              Linear(32 -> 1)
                       |
                       v
                  prediction
```

So it's not a `38->32->1` network. Each layer has **4 separate linear transforms** (the 4 gates), plus memory mechanics, and it runs this **at every timestep**. That's what makes it fundamentally different from an MLP.

---

## Deep Dive: Sigmoid vs Tanh

Sigmoid and tanh have different output ranges and serve different purposes:

- **Sigmoid** (0 to 1) — used for gates. It's a "how much" dial. 0 means block everything, 1 means let everything through. Perfect for controlling information flow.
- **Tanh** (-1 to 1) — used for values. The candidate values and the cell state readout use tanh because actual data can be negative. Sigmoid can't represent negative values.

Where each is used:

```
f = sigmoid(...)     gate: how much to forget (0-1)
i = sigmoid(...)     gate: how much to write (0-1)
candidate = tanh(...) value: what to write (-1 to 1)
o = sigmoid(...)     gate: how much to expose (0-1)
h_t = o * tanh(c_t)  value: bounded output (-1 to 1)
```

---

## Deep Dive: Training vs Streaming

### Training (Offline)

You take historical data (e.g., 1000 price points) and slide a window across it to create overlapping sequences:

```
prices: [p1, p2, p3, p4, p5, p6, p7, ...]

sequence 1: [p1, p2, p3, p4, p5]  → target: p6
sequence 2: [p2, p3, p4, p5, p6]  → target: p7
sequence 3: [p3, p4, p5, p6, p7]  → target: p8
...
```

Each sequence goes through feature extraction to become `(seq_len, 6)`. These get grouped into **batches** and fed to the training loop.

`batch_size` is the number of sequences **per batch**, not the number of batches. If you have 800 sequences and `batch_size=8`, that's 100 batches.

One **epoch** = one pass through all the data:

```
epoch
├── batch 1:  sequences 1-8
├── batch 2:  sequences 9-16
├── batch 3:  sequences 17-24
├── ...
└── batch 100: sequences 793-800
```

Each batch has shape `(batch_size, seq_len, 6)`. The LSTM processes all sequences in a batch independently but simultaneously — matrix multiplication on a batch is much faster than looping one sequence at a time. After all 100 batches, the epoch is done. Then you repeat for the next epoch.

### Live Prediction (Streaming)

The buffer holds the latest window (e.g., last 50 points). Each new price pushes the window forward by one. Feature extraction runs on the window, producing one `(seq_len, 6)` array. That goes to `execute` — one sequence, one prediction. No batches, no epochs.

```python
# Model.execute()
x = torch.from_numpy(features)          # shape (seq_len, 6)
x = x.unsqueeze(0)                      # shape (1, seq_len, 6) — fake batch of 1
output = self.model(x)                   # shape (1, 1)
return output.squeeze(0).numpy()         # shape (1,) — remove the fake batch dim
```

The LSTM always expects a batch dimension. `unsqueeze(0)` adds it, `squeeze(0)` removes it.

### batch_first=True

```python
nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
```

Without `batch_first=True`, `nn.LSTM` expects shape `(seq_len, batch, features)`. With it, the shape is `(batch, seq_len, features)`. It's just a convention choice — batch-first is more intuitive since most code thinks in terms of `(batch, ...)`. Either way works, the batch dimension exists regardless of its size.

### self.lstm() Return Values

`self.lstm(x)` returns two things:

1. **`lstm_out`** — the last layer's hidden state at **every** timestep, stacked. Shape `(batch, seq_len, hidden_dim)`.
2. **`(h_final, c_final)`** — the final hidden state and cell state for **all** layers. We discard this with `_` because we already have the last timestep's hidden state in `lstm_out`.

```python
lstm_out, _ = self.lstm(x)       # lstm_out: (batch, seq_len, 32)
lstm_out[:, -1, :]               # (batch, 32) — just the last timestep
```

For a single streaming prediction with `batch=1` and `seq_len=10`:

```
lstm_out shape: (1, 10, 32)     — h2 at all 10 timesteps
lstm_out[:, -1, :] shape: (1, 32) — h2 at timestep 10 only
self.fc(...) shape: (1, 1)       — one prediction
```

---

## History

The LSTM was not discovered by luck. Hochreiter and Schmidthuber published it in 1997, specifically to solve the vanishing gradient problem they had mathematically analyzed in plain RNNs. The forget gate was added later by Gers et al. (2000). The architecture was engineered, not found by accident — though there's been empirical work since then testing which gates are actually necessary (GRU, for example, simplifies to two gates and works nearly as well).
