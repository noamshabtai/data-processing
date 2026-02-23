import torch

from . import model


class Trainer:
    def __init__(self, **kwargs):
        self.epochs = kwargs.get("epochs")
        self.learning_rate = kwargs.get("learning_rate")
        self.batch_size = kwargs.get("batch_size")
        self.data_shuffle = kwargs.get("data_shuffle", True)
        self.model = model.Model(**kwargs["network"])

    def execute(self, features):
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            output = self.model(x)
            return output.squeeze(0).numpy()

    def backward(self, data, targets):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        x = torch.from_numpy(data)
        y = torch.from_numpy(targets)
        batch_size = self.batch_size if self.batch_size is not None else len(x)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=self.data_shuffle
        )
        epoch_losses = []
        for _ in range(self.epochs):
            batch_losses = []
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(sum(batch_losses) / len(batch_losses))
        return epoch_losses

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
