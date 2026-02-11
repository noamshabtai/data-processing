# LSTM & Project Walkthrough Conversation

## How data-processing works, step by step

### The Big Picture

This project takes **live stock prices**, extracts **meaningful patterns** from them, and feeds those patterns into an **LSTM neural network** to predict future values. Think of it as a pipeline:

```
Raw stock prices -> Signal features -> Neural network -> Prediction
```

---

### Step 1: Fetching Stock Data (`data-fetcher`)

Everything starts with real market data. `Fetcher` wraps the Yahoo Finance API:

```python
# data-fetcher/src/data_fetcher/fetcher.py
fetcher = Fetcher(symbol="AAPL", period="7d", interval="1h")
df = fetcher.fetch_historical()  # DataFrame with Open, High, Low, Close, Volume
```

This gives you a table of price candles — each row is one hour of AAPL trading. The key column you care about downstream is **Close** (the closing price at each interval).

`fetch_latest()` grabs the most recent 1-minute candle for live use.

---

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

---

### Step 3: Feature Extraction (`feature-extraction`)

Raw prices aren't great direct input for a neural network. Instead, you extract **features** — compact numbers that describe *what the signal looks like*.

This project uses two extractors:

#### TrendExtractor — "Where is the price going?"

```python
# feature-extraction/src/feature_extraction/trend.py
def extract(self, data):
    moving_avg = np.convolve(data, np.ones(self.window_size) / self.window_size, mode="valid")
    slope = np.gradient(moving_avg)
    return np.array([moving_avg[-1], slope[-1], np.std(slope)])
```

Three values:
1. **`moving_avg[-1]`** — the current smoothed price (filters out noise)
2. **`slope[-1]`** — is the price going up or down *right now*?
3. **`np.std(slope)`** — how volatile is the trend? (stable vs. erratic)

#### FFTExtractor — "Are there repeating patterns?"

```python
# feature-extraction/src/feature_extraction/fft.py
def extract(self, data):
    fft_result = np.fft.rfft(data)
    magnitudes = np.abs(fft_result)
    dominant_freq_idx = np.argmax(magnitudes[1:]) + 1
    return np.array([dominant_freq_idx, magnitudes[dominant_freq_idx], np.mean(magnitudes)])
```

FFT decomposes a signal into frequencies (like a prism splitting light into colors). Three values:
1. **`dominant_freq_idx`** — which frequency is strongest (is there a cycle?)
2. **`magnitudes[dominant_freq_idx]`** — how strong is that cycle?
3. **`np.mean(magnitudes)`** — overall signal energy

#### Pipeline — glue them together

```python
# Pipeline.extract() output: [ma, slope, slope_std, freq_idx, freq_mag, mean_mag]
```

You now have a **6-element feature vector** for each window of data.

---

### Step 4: LSTM Prediction (`model`)

This is the PyTorch part. Here's how the LSTM works:

```python
# model/src/model/lstm.py
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)          # process the sequence
        output = self.fc(lstm_out[:, -1, :]) # take the last timestep
        return output
```

**PyTorch crash course for this code:**

- `nn.Module` — base class for all neural network layers in PyTorch. You define layers in `__init__` and the computation in `forward`.
- `nn.LSTM(input_dim=6, hidden_dim=32, num_layers=2, batch_first=True)` — a 2-layer LSTM. Each timestep receives 6 features. Internally it maintains a 32-dimensional hidden state that carries memory across timesteps. `batch_first=True` means the input shape is `(batch, sequence_length, 6)`.
- `nn.Linear(32, 1)` — a fully connected layer that maps the 32-dim hidden state to 1 output value (the prediction).
- `lstm_out[:, -1, :]` — from the output of all timesteps, take only the **last** one. The idea: the final hidden state has "seen" the entire sequence and encodes the most up-to-date understanding.

The `Predictor` class wraps this for inference:

```python
# model/src/model/predictor.py
def predict(self, features):
    with torch.no_grad():                        # no gradient tracking (inference only)
        x = torch.from_numpy(features).unsqueeze(0)  # numpy -> tensor, add batch dim
        output = self.model(x)
        return output.numpy().squeeze()          # tensor -> numpy, remove batch dim
```

Key PyTorch concepts here:
- **`torch.no_grad()`** — tells PyTorch "don't track gradients." During training you need gradients for backpropagation, but during inference it's wasted memory/computation.
- **`unsqueeze(0)`** — adds a batch dimension. The model expects `(batch, seq, features)` but you're feeding one sample, so you go from shape `(seq, 6)` to `(1, seq, 6)`.
- **`squeeze()`** — removes that batch dimension from the output.

---

### Step 5: Putting It All Together (`finance-demo`)

The `System` class wires the modules:

```python
# finance-demo/src/finance_demo/system.py
class System(system.system.System):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modules["features"] = Pipeline(**kwargs.get("features", {}))
        self.modules["predictor"] = Predictor(**kwargs.get("predictor", {}))

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

---

### Step 6: CLI (`stock-analyzer`)

The entry point for users:

```bash
stock-analyzer train --symbol AAPL --period 1y --epochs 10 --output model.pt
stock-analyzer predict --symbol AAPL --model model.pt
```

This is the thin command-line layer that would orchestrate training and live prediction using all the modules above.

---

### The Full Data Flow

```
AAPL price tick (yfinance)
    │
    ▼
Input Buffer (keeps last 50 points)
    │
    ▼
┌─────────────────────┐
│  TrendExtractor     │──→ [moving_avg, slope, slope_std]
│  FFTExtractor       │──→ [dom_freq, dom_mag, mean_mag]
└─────────────────────┘
    │
    ▼
Feature vector: [6 values]
    │
    ▼
LSTM (2 layers, hidden_dim=32)
    │
    ▼
Prediction (1 value)
```

### Testing

Every module has YAML-driven parametrized tests. Instead of hardcoding test params in Python, they're defined in `tests/config/*.yaml` and the `parametrize-tests` utility generates pytest fixtures from them. This makes it easy to add test cases without touching Python code.

---

## Q&A: nn.Linear, Activations, and Hidden Layers

### What is `nn.Linear`?

`nn.Linear` is NOT an activation. It's a **fully connected (dense) layer** — just a matrix multiplication plus a bias:

```
output = input @ W + b
```

Where `W` is a weight matrix and `b` is a bias vector. That's it. Purely linear math, no activation function.

An **activation** would be something like `nn.ReLU()`, `nn.Sigmoid()`, or `nn.Tanh()` — the nonlinear function applied *after* a linear layer.

### How many hidden layers does this model have?

Looking at the current code:

```python
self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
self.fc = nn.Linear(hidden_dim, output_dim)  # this is the OUTPUT layer
```

There are **zero hidden fully-connected layers**. The LSTM does the heavy lifting (2 stacked LSTM layers internally), and then `nn.Linear(32, 1)` is just the **output layer** — it maps the 32-dim LSTM output directly to 1 prediction value. No activation, no hidden dense layers in between.

Think of it as:

```
LSTM (2 recurrent layers, 32-dim state) → Linear(32 → 1) → done
```

### What would 2 hidden layers look like?

Current model vs. a version with 2 hidden fully-connected layers:

```python
# CURRENT — no hidden FC layers
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)       # 32 → 1 (output)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output
```

```python
# WITH 2 HIDDEN LAYERS
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)              # 32 → 64 (hidden 1)
        self.relu1 = nn.ReLU()                             # activation
        self.fc2 = nn.Linear(64, 32)                       # 64 → 32 (hidden 2)
        self.relu2 = nn.ReLU()                             # activation
        self.fc3 = nn.Linear(32, output_dim)               # 32 → 1  (output)

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
Linear  →  Activation  →  Linear  →  Activation  →  ...  →  Linear (output)
```

**Why activations between layers?** Without them, stacking linear layers is pointless — `Linear(Linear(x))` is still just a linear function. The activation (ReLU, etc.) adds nonlinearity, which is what lets the network learn complex patterns.

**Why no activation on the output layer?** Because this is a regression task (predicting a price). You want the output to be any real number, not squashed through ReLU (which clips negatives) or sigmoid (which squashes to 0-1).

### Visual comparison

```
CURRENT:
  LSTM[2 layers] → Linear(32→1) → prediction

WITH 2 HIDDEN:
  LSTM[2 layers] → Linear(32→64) → ReLU → Linear(64→32) → ReLU → Linear(32→1) → prediction
```

The current design keeps it simple — the LSTM layers themselves already have a lot of capacity, so a single output projection is often enough.

---

## Q&A: MLP vs LSTM

### MLP (Multi-Layer Perceptron)

What you might picture with 2 hidden layers — that's an MLP:

```
input → Linear → ReLU → Linear → ReLU → Linear → output
```

An MLP sees all its input **at once**, as a flat vector. It has no concept of order or time. If you fed it 50 stock prices, it treats them as 50 independent numbers — it doesn't know that price #1 came before price #2.

### LSTM (Long Short-Term Memory)

An LSTM processes data **one timestep at a time**, carrying a memory (hidden state) forward:

```
price₁ → [LSTM cell] → hidden₁
                          ↓
price₂ → [LSTM cell] → hidden₂
                          ↓
price₃ → [LSTM cell] → hidden₃
                          ↓
                     ... and so on
                          ↓
price₅₀ → [LSTM cell] → hidden₅₀ → Linear → prediction
```

At each step, the LSTM cell decides:
- **What to forget** from its memory (old trends that faded)
- **What to remember** from the new input (a sudden price spike)
- **What to output** as the current hidden state

### Side by side

| | MLP | LSTM |
|---|---|---|
| Sees input as | Flat vector (all at once) | Ordered sequence (step by step) |
| Has memory | No | Yes — hidden state carries across steps |
| Good for | Tabular data, classification | Time series, sequences, text |
| Knows order | No — shuffling inputs changes nothing | Yes — order is everything |

### What this project actually does

```
          LSTM part                    MLP part (sort of)
   ┌─────────────────────┐         ┌──────────────┐
   │ step 1 → cell → h₁  │         │              │
   │ step 2 → cell → h₂  │         │              │
   │ ...                  │  ──→    │ Linear(32→1) │  ──→  prediction
   │ step N → cell → hₙ  │  last   │              │
   │   (2 stacked layers) │  step   │              │
   └─────────────────────┘         └──────────────┘
```

The LSTM processes the sequence and builds up understanding over time. Then the final hidden state (`h_N`) gets passed through a single `Linear` layer to produce the prediction. That single `Linear` at the end is technically a 0-hidden-layer MLP — just an output projection.

So this model is: **LSTM for sequence processing** + **a tiny MLP head for output**. Not one or the other — they work together.

---

## Deep Dive: Inside the LSTM Cell

### Inside a single LSTM cell

An LSTM cell has **4 gates** — small neural networks that control information flow. At each timestep, the cell receives the current input `x_t` and the previous hidden state `h_{t-1}`, and maintains a **cell state** `C` (the long-term memory).

```
                    ┌─────────────────────────────────────┐
                    │           LSTM Cell                  │
                    │                                     │
  C_{t-1} ────────►│──► [forget gate] ──► throw away     │────► C_t
  (old memory)      │                      old memory      │   (new memory)
                    │──► [input gate]  ──► what to add     │
                    │──► [candidate]   ──► new values      │
                    │──► [output gate] ──► what to expose  │
  h_{t-1} ────────►│                                     │────► h_t
  (old hidden)      │                                     │   (new hidden)
                    │                                     │
  x_t ────────────►│                                     │
  (current input)   └─────────────────────────────────────┘
```

The 4 gates, step by step:

**1. Forget gate** — "what should I erase from memory?"

```
f = sigmoid(W_f · [h_{t-1}, x_t] + b_f)
```

Outputs values between 0 and 1 for each element of the cell state. 0 = completely forget, 1 = fully keep. Example: if the trend reversed, forget the old trend direction.

**2. Input gate** — "what new information is worth storing?"

```
i = sigmoid(W_i · [h_{t-1}, x_t] + b_i)
```

Also 0-1 values. Decides *which* parts of the memory to update.

**3. Candidate values** — "what are the new values to potentially store?"

```
C̃ = tanh(W_c · [h_{t-1}, x_t] + b_c)
```

These are the proposed new values (between -1 and 1).

**4. Update the cell state** — apply forget and input:

```
C_t = f * C_{t-1} + i * C̃
      ^^^^^^^^^^^   ^^^^^^^
      keep this     add this
```

This is the key insight of LSTMs. The cell state flows through time with only **element-wise** operations — no matrix multiplications. This is why LSTMs can remember things over long sequences (solves the vanishing gradient problem that plain RNNs have).

**5. Output gate** — "what part of my memory should I expose?"

```
o = sigmoid(W_o · [h_{t-1}, x_t] + b_o)
h_t = o * tanh(C_t)
```

The cell may know a lot, but only exposes the relevant part as its hidden state `h_t`.

---

### So each gate is itself a little neural network?

Yes. Each gate is:

```
Linear(input_dim + hidden_dim → hidden_dim) → sigmoid (or tanh)
```

That's why LSTMs have so many parameters. With `input_dim=6` and `hidden_dim=32`, a single LSTM layer has 4 gates, each with a weight matrix of size `(6+32, 32)`. That's `4 × (38 × 32 + 32 bias) = 4 × 1248 = 4992` parameters in one layer.

---

### What "2 stacked layers" means

```python
self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
```

`num_layers=2` means two LSTM layers **stacked vertically**. The output sequence of layer 1 becomes the input sequence of layer 2:

```
                    Layer 1                    Layer 2
                  (input → 32)               (32 → 32)

timestep 1:   x₁ → [LSTM cell] → h¹₁   →   [LSTM cell] → h²₁
                        ↓                        ↓
timestep 2:   x₂ → [LSTM cell] → h¹₂   →   [LSTM cell] → h²₂
                        ↓                        ↓
timestep 3:   x₃ → [LSTM cell] → h¹₃   →   [LSTM cell] → h²₃
                        ↓                        ↓
              ...      ...                      ...
                        ↓                        ↓
timestep N:   xₙ → [LSTM cell] → h¹ₙ   →   [LSTM cell] → h²ₙ
                                                              ↓
                                                        Linear(32→1)
                                                              ↓
                                                         prediction
```

- **Layer 1** sees the raw features (6 values per timestep) and produces a 32-dim hidden state at each step
- **Layer 2** sees Layer 1's hidden states (32 values per timestep) and produces its own 32-dim hidden states
- Only the **last hidden state** of Layer 2 (`h²ₙ`) goes to the output Linear layer

### Are they "hidden" layers?

Yes, in the sense that they're between input and output and you never see their values directly. But they're not the same as MLP hidden layers:

| | MLP hidden layer | LSTM stacked layer |
|---|---|---|
| What it is | `Linear + activation` | Full LSTM cell (4 gates, cell state, hidden state) |
| Parameters | One weight matrix | Four weight matrices + biases |
| Has memory | No | Yes — carries state across timesteps |
| Sees data | Once | Once per timestep, sequentially |

### Why stack them?

Same intuition as MLP — more layers = more abstraction:

- **Layer 1** might learn low-level patterns: "price is rising", "there's a dip"
- **Layer 2** takes those patterns and learns higher-level ones: "this looks like a reversal", "this dip is within a larger uptrend"

In this project's code, `num_layers=2` is set via the YAML config:

```yaml
predictor:
  input_dim: 6       # 6 features from the extraction pipeline
  hidden_dim: 32     # each LSTM layer has 32-dim hidden state
  output_dim: 1      # predict one value
  num_layers: 2      # two stacked LSTM layers
```

---

## Deep Dive: Element-wise Operations and Data Flow

### What is `f * something`?

`*` here means **element-wise multiplication**, not matrix multiplication. Every value in `f` multiplies the corresponding value in `C_{t-1}`, one by one.

`f` is a vector of 32 values, each between 0 and 1 (because sigmoid). `C_{t-1}` is also a vector of 32 values. So:

```
f         = [0.9,  0.1,  0.7,  ...]    (32 values, each 0-1)
C_{t-1}   = [3.2, -1.5,  0.8,  ...]    (32 values, the memory)

f * C_{t-1} = [2.88, -0.15, 0.56, ...]  (keep 90% of slot 1, erase 90% of slot 2, etc.)
```

Same with `i * C̃` — the input gate `i` controls how much of each proposed new value actually gets written to memory.

---

### How is `C_t` used?

`C_t` is used in the very next line to produce the hidden state:

```
C_t = f * C_{t-1} + i * C̃       ← update memory
h_t = o * tanh(C_t)              ← read from memory
```

And then `C_t` gets passed to the **next timestep** as `C_{t-1}`. It's a chain:

```
C_0 → C_1 → C_2 → C_3 → ... → C_N
```

The cell state is the **internal memory** that flows across time. The hidden state `h_t` is a filtered view of it — what the cell chooses to expose.

---

### What is `h_t`?

`h_t` is a single vector of 32 values. It's the output of the LSTM cell at timestep `t`. Not "all outputs stacked" — just one timestep's output.

Over a whole sequence, you get a **series** of hidden states:

```
h_1, h_2, h_3, ..., h_N     (each is 32 values)
```

PyTorch returns all of them stacked as `lstm_out` with shape `(batch, N, 32)`. Then the code takes only the last one:

```python
lstm_out[:, -1, :]   # shape: (batch, 32) — just h_N
```

---

### What does "stacking h with inputs" mean?

Inside the LSTM cell, the notation `[h_{t-1}, x_t]` means **concatenation**. If `h` is 32 values and `x` is 6 values, you concatenate them into one 38-value vector:

```
h_{t-1} = [h₁, h₂, ..., h₃₂]         (32 values)
x_t     = [x₁, x₂, ..., x₆]          (6 values)

[h_{t-1}, x_t] = [h₁, h₂, ..., h₃₂, x₁, ..., x₆]   (38 values)
```

Then each gate multiplies this 38-value vector by a weight matrix:

```
f = sigmoid(W_f · [38 values] + b_f)
              ^^^^
         W_f is a (38 × 32) matrix
```

So yes — **within Layer 1**, each gate is essentially a `Linear(38 → 32) + sigmoid`.

---

### What about Layer 2?

Layer 2 does **not** see the original 6 input features. It only sees Layer 1's hidden states.

Layer 2 concatenates **its own** previous hidden state with Layer 1's output:

```
Layer 1, timestep t:  [h¹_{t-1}(32), x_t(6)]     → concat → 38 values → gates → h¹_t (32)
                                                                                    ↓
Layer 2, timestep t:  [h²_{t-1}(32), h¹_t(32)]    → concat → 64 values → gates → h²_t (32)
```

So the weight matrices are different sizes per layer:

```
Layer 1 gates: W is (6 + 32)  × 32 = 38 × 32   (input + own hidden)
Layer 2 gates: W is (32 + 32) × 32 = 64 × 32   (layer 1 output + own hidden)
```

---

### The full picture for one timestep

```
x_t (6 values)
  │
  ▼
┌──────────────────────────────────────────────────┐
│ LSTM Layer 1                                     │
│                                                  │
│  concat [h¹_{t-1}, x_t] → 38 values             │
│    ├─ forget gate: Linear(38→32) + sigmoid → f   │
│    ├─ input gate:  Linear(38→32) + sigmoid → i   │
│    ├─ candidate:   Linear(38→32) + tanh    → C̃   │
│    └─ output gate: Linear(38→32) + sigmoid → o   │
│                                                  │
│  C¹_t = f * C¹_{t-1} + i * C̃                     │
│  h¹_t = o * tanh(C¹_t)                           │
└──────────────────────┬───────────────────────────┘
                       │ h¹_t (32 values)
                       ▼
┌──────────────────────────────────────────────────┐
│ LSTM Layer 2                                     │
│                                                  │
│  concat [h²_{t-1}, h¹_t] → 64 values            │
│    ├─ forget gate: Linear(64→32) + sigmoid → f   │
│    ├─ input gate:  Linear(64→32) + sigmoid → i   │
│    ├─ candidate:   Linear(64→32) + tanh    → C̃   │
│    └─ output gate: Linear(64→32) + sigmoid → o   │
│                                                  │
│  C²_t = f * C²_{t-1} + i * C̃                     │
│  h²_t = o * tanh(C²_t)                           │
└──────────────────────┬───────────────────────────┘
                       │ h²_t (32 values)
                       ▼
              Linear(32 → 1)
                       │
                       ▼
                  prediction
```

So it's not a `38→32→1` network. Each layer has **4 separate linear transforms** (the 4 gates), plus memory mechanics, and it runs this **at every timestep**. That's what makes it fundamentally different from an MLP.
