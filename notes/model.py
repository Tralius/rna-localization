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

    @abstractmethod
    def fit_and_evaluate(self, train_data, eval_data, **kwargs):
        pass

    @abstractmethod
    def predict(self, data, **kwargs):
        pass

    @abstractmethod
    def print_model(self, path):
        pass

    @abstractmethod
    def save_model(self, path):
        pass