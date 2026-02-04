import torch

from . import lstm


class Predictor:
    def __init__(self, **kwargs):
        self.model = lstm.LSTM(**kwargs)
        self.model.eval()

    def predict(self, features):
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            output = self.model(x)
            return output.numpy().squeeze()

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
