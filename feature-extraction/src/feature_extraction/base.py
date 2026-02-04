import abc


class BaseExtractor(abc.ABC):
    def __init__(self, **kwargs):
        self.window_size = kwargs.get("window_size", 20)

    @abc.abstractmethod
    def extract(self, data):
        pass
