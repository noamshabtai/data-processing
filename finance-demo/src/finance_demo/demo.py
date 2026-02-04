import time

import activator.base_demo

from . import system


class Activator(activator.base_demo.Activator):
    def __init__(self, **kwargs):
        super().__init__(activated_system=system.System, **kwargs)
        self.poll_interval = kwargs.get("poll_interval", 60)

    def execute(self):
        self.running = True
        while self.running:
            frame = self._fetch_latest()
            if frame is not None:
                prediction = self.process_frame(frame)
                if prediction is not None:
                    self._log_prediction(frame, prediction)
            time.sleep(self.poll_interval)
        self.completed = True

    def cleanup(self):
        self.running = False

    def _fetch_latest(self):
        return None

    def _log_prediction(self, frame, prediction):
        print(f"Prediction: {prediction}")
