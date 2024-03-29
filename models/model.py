from abc import ABC
from keras.utils import plot_model
from dataloaders.GeneDataLoader import GeneDataLoader


class Model(ABC):
    def __init__(self, **kwargs) -> None:
        self.model = None

    def fit(self, train_data, params_dataLoader, params_train):
        if params_train is None:
            Warning('evalutation with default parameters')
            params_train = {}
        if params_dataLoader is None:
            Warning('data Loader uses default arguments')
            params_dataLoader = {}
        if ('i' in self.architecture and not params_dataLoader.get('m6a')) or (params_dataLoader.get('m6a') and not 'i' in self.architecture):
            ValueError('m6a input requires concatination')
        return self.model.fit(GeneDataLoader(train_data, **params_dataLoader), **params_train)

    def evaluate(self, eval_data, params_dataLoader, params_eval):
        if params_eval is None:
            Warning('evalutation with default parameters')
            params_eval = {}
        if params_dataLoader is None:
            Warning('data Loader uses default arguments')
            params_dataLoader = {}
        return self.model.evaluate(GeneDataLoader(eval_data, **params_dataLoader), **params_eval)

    def fit_and_evaluate(self, train_data, eval_data, callback, params_train_dataLoader, params_eval_dataLoader, params_train):
        if params_train_dataLoader is None:
            Warning('data Loader uses default arguments')
            params_train_dataLoader = {}
        if params_eval_dataLoader is None:
            Warning('data Loader uses default arguments')
            params_eval_dataLoader = {}
        if params_train is None:
            Warning('evalutation with default parameters')
            params_train = {}

        train_dataLoader = GeneDataLoader(train_data, **params_train_dataLoader)
        eval_dataLoader = GeneDataLoader(eval_data, shuffle=False, **params_eval_dataLoader)
        return self.model.fit(train_dataLoader, callbacks=callback, validation_data=eval_dataLoader, **params_train)

    def predict(self, pred_data, params_dataLoader, params_predict):
        if params_predict is None:
            Warning('evalutation with default parameters')
            params_predict = {}
        if params_dataLoader is None:
            Warning('data Loader uses default arguments')
            params_dataLoader = {}
        return self.model.predict(GeneDataLoader(pred_data, **params_dataLoader), **params_predict)

    def summary(self):
        return self.model.summary()

    def print_model(self, path):
        if path is None:
            return plot_model(self.model, show_shapes=True)
        return plot_model(self.model, path, show_shapes=True)

    def save_model(self, path):
        self.model.save(path)
        
    def load_weights(self, path):
        return self.model.load_weights(path)
