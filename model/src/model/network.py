from torch import nn


class Network(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=kwargs["input_dim"],
            hidden_size=kwargs["hidden_dim"],
            num_layers=kwargs.get("num_layers", 2),
            batch_first=True,
        )
        self.fc = nn.Linear(
            in_features=kwargs["hidden_dim"],
            out_features=kwargs.get("output_dim", 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
