import feature_extraction.feature_extraction
import system.system

import model.trainer


class System(system.system.System):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modules["features"] = feature_extraction.feature_extraction.FeatureExtraction(**kwargs.get("features", {}))
        self.modules["predictor"] = model.trainer.Trainer(**kwargs.get("predictor", {}))

    def connect(self, module):
        match module:
            case "features":
                self.inputs[module] = dict(data=self.input_buffer.get_window())
            case "predictor":
                self.inputs[module] = dict(features=self.outputs["features"])
