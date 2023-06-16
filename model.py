from abc import ABC, abstractmethod
from GeneDataLoader import GeneDataLoader
import pandas as pd

class Model(ABC):
    def __init__(self, train_data: pd.DataFrame, validation_data: pd.DataFrame = None, **kwargs) -> None:
        batch_size_train = kwargs.pop('batch_size_train')
        shuffle_batch_train = kwargs.pop('shuffle_batch_train')
        self.train_data_loader = GeneDataLoader(train_data, batch_size=batch_size_train, shuffle=shuffle_batch_train)
        if validation_data is not None:
            batch_size_valid = kwargs.pop('batch_size_valid')
            shuffle_batch_valid = kwargs.pop('shuffle_batch_valid')
            self.validation_data_loader = GeneDataLoader(validation_data, batch_size=batch_size_valid, shuffle=shuffle_batch_valid)

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass