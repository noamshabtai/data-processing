import feature_extraction.feature_extraction
import system.system
import torch

import model.model


class System(system.system.System):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modules["feature_extraction"] = feature_extraction.feature_extraction.FeatureExtraction(
            **kwargs["feature_extraction"]
        )
        self.modules["model"] = model.model.Model(**kwargs["model"])
        self.modules["model"].execute = self.modules["model"].forward

    def connect(self, module):
        match module:
            case "feature_extraction":
                self.inputs[module] = {"data": self.input_buffer.buffer}
            case "model":
                self.inputs[module] = {
                    "x": torch.from_numpy(self.outputs["feature_extraction"]).reshape(1, 1, -1).float()
                }
