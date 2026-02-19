import torch
from torch import nn

from . import network


class Model:
    def __init__(self, **kwargs):
        self.model = network.Network(**kwargs)

    def execute(self, features):
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            output = self.model(x)
            return output.squeeze(0).numpy()

    def backward(self, data, targets, epochs, lr, batch_size=None):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        x = torch.from_numpy(data)
        y = torch.from_numpy(targets)
        if batch_size is None:
            batch_size = len(x)
        epoch_loss = None
        for _ in range(epochs):
            batch_losses = []
            for x_batch, y_batch in zip(x.split(batch_size), y.split(batch_size)):
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_loss = sum(batch_losses) / len(batch_losses)
        return epoch_loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
