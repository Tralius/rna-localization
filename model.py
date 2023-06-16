from abc import ABC, abstractmethod
import pandas as pd

class Model(ABC):
    def __init__(self, train_data: pd.DataFrame, validation_data: pd.DataFrame = None, **kwargs) -> None:
        pass

    @abstractmethod
    def fit(self, train_data, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, eval_data, **kwargs):
        pass