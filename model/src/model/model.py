import numpy as np
import torch
from torch import nn


class _LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class Model:
    def __init__(self, **kwargs):
        self.model = _LSTM(
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs["hidden_dim"],
            output_dim=kwargs.get("output_dim", 1),
            num_layers=kwargs.get("num_layers", 2),
        )
        self.model.eval()

    def execute(self, features):
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            output = self.model(x)
            return output.squeeze(0).numpy()

    def backward(self, data, targets, epochs, lr):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        x = torch.from_numpy(data)
        y = torch.from_numpy(targets)
        loss_value = None
        for _ in range(epochs):
            optimizer.zero_grad()
            output = self.model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
        self.model.eval()
        return loss_value

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
