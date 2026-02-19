from torch import nn


class Network(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(kwargs["input_dim"], kwargs["hidden_dim"], kwargs.get("num_layers", 2), batch_first=True)
        self.fc = nn.Linear(kwargs["hidden_dim"], kwargs.get("output_dim", 1))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
