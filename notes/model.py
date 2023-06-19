from abc import ABC, abstractmethod
import pandas as pd


class Model(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def fit(self, train_data, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, eval_data, **kwargs):
        pass
