from abc import ABC, abstractmethod
import pandas as pd
from dataloaders.GeneDataLoader import GeneDataLoader

class Func_Model(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    def fit(self, train_data, params_dataLoader, params_train):
        if params_train is None:
            Warning('evalutation with default parameters')
            params_train = {}
        if params_dataLoader is None:
            Warning('data Loader uses default arguments')
            params_dataLoader = {}
        return self.model.fit(GeneDataLoader(train_data, **params_dataLoader), **params_train)

    def evaluate(self, eval_data, params_dataLoader, params_eval):
        if params_eval is None:
            Warning('evalutation with default parameters')
            params_eval = {}
        if params_dataLoader is None:
            Warning('data Loader uses default arguments')
            params_dataLoader = {}
        return self.model.evaluate(GeneDataLoader(eval_data, **params_dataLoader), **params_eval)

    def predict(self, pred_data, params_dataLoader, params_predict):
        if params_predict is None:
            Warning('evalutation with default parameters')
            params_predict = {}
        if params_dataLoader is None:
            Warning('data Loader uses default arguments')
            params_dataLoader = {}
        return self.model.predict(GeneDataLoader(pred_data, **params_dataLoader), **params_predict)
