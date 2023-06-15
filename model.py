from abc import ABC, abstractmethod
from GeneDataLoader import GeneDataLoader

class Model(ABC):
    def __init__(self, data, **kwargs) -> None:
        self.loader = GeneDataLoader(data_table=data, **kwargs)
        self.model = None

    @abstractmethod
    def train(loss, **kwargs):
        pass

    @abstractmethod
    def predict(data, **kwargs):
        pass