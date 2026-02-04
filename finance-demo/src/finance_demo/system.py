import feature_extraction.pipeline
import system.system

import model.predictor


class System(system.system.System):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modules["features"] = feature_extraction.pipeline.Pipeline(**kwargs.get("features", {}))
        self.modules["predictor"] = model.predictor.Predictor(**kwargs.get("predictor", {}))

    def connect(self, module):
        match module:
            case "features":
                self.inputs[module] = dict(data=self.input_buffer.get_window())
            case "predictor":
                self.inputs[module] = dict(features=self.outputs["features"])
