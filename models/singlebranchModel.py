from typing import Dict, Tuple, List
import keras
import pandas as pd
from models import utils, Model
from keras.optimizers import Adam, SGD
from keras.metrics import CategoricalCrossentropy, KLDivergence
from keras.losses import CategoricalCrossentropy
from keras.callbacks import ModelCheckpoint
from metrics import pearson

class CNN(Model):
    """
    Architecture:
    a: Multihead-Attention layer (for singlehead set heads=1)
    c: 1D Convolution
    d: Dropout layer
    e: Dense layer
    f: Flatten layer
    p: 1D Max-Pooling layer
    """
    def __init__(self,
                 input_size: Tuple,
                 optimizer = keras.optimizers.Adam(),
                 loss = CategoricalCrossentropy(),
                 metrics = ['accuracy', KLDivergence(name="kullback_leibler_divergence"), pearson],
                 params_model: Dict[str, List[Dict]] = None,
                 compile: Dict = None,
                 checkpoint_filepath = None) -> None:


        super().__init__()
        
        if params_model is None:
            params_model = {}
        if compile is None:
            compile = {}
        
        input_lay = keras.Input(shape=input_size)

        architecture = list(params_model.get('architecture'))
        utils.check_params(params_model)

        index = {}
        for key in params_model.keys():
            index[key] = 0
            
        arch = []

        for k, j in enumerate(list(architecture)):
            if k == 0:
                arch, index = utils.add_layer(j, input_lay, index, params_model, arch)
            else:
                arch, index = utils.add_layer(j, arch[len(arch)-1], index, params_model, arch)
        
        self.model = keras.Model(inputs=input_lay, outputs=arch[-1])

        if "optimizer" not in params_model.keys():
            optimizer = 'adam'
        else:
            optimizer = params_model['optimizer']
        if 'learning_rate' not in params_model.keys():
            learning_rate = None
        else:
            learning_rate = float(params_model['learning_rate'])
        optimizer = utils.set_optimizer(optimizer, learning_rate)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **compile)

    def fit(self, train_data: pd.DataFrame, params_dataLoader: Dict = None, params_train: Dict = None):
        return super().fit(train_data, params_dataLoader, params_train)
    
    def evaluate(self, eval_data: pd.DataFrame, params_dataLoader: Dict = None, params_eval: Dict = None):
        return super().evaluate(eval_data, params_dataLoader, params_eval)

    def fit_and_evaluate(self, train_data, eval_data, callback: List[keras.callbacks.Callback] = None,
                         params_train_dataLoader: Dict = None,
                         params_eval_dataLoader: Dict = None,
                         params_train: Dict = None):
        return super().fit_and_evaluate(train_data, eval_data, callback,
                                        params_train_dataLoader,
                                        params_eval_dataLoader,
                                        params_train)

    def predict(self, pred_data, params_dataLoader: Dict = None, params_predict: Dict = None):
        return super().predict(pred_data, params_dataLoader, params_predict)

    def summary(self):
        return super().summary()

    def print_model(self, path=None):
        return super().print_model(path)

    def save_model(self, path):
        super().save_model(path)

    def load_weights(self, path):
        return super().load_weights(path)
