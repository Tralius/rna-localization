import keras
import numpy as np
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, MultiHeadAttention, Reshape, LeakyReLU, \
    BatchNormalization, Concatenate, add
from models import Func_Model, utils
from typing import Dict, Tuple, List
from collections import Counter
from dataloaders import GeneDataLoader


class MultiBranch(Func_Model):
    """

    TODO: refine docs
    ...

    Architecture:
    a: Multihead-Attention layer (for singlehead set heads=1)
    c: 1D Convolution
    d: Dropout layer
    e: Dense layer
    f: Flatten layer
    p: 1D Max-Pooling layer
    r: Reshape layer
    """

    def __init__(self,
                 input_size: Tuple,
                 number_branches: int,
                 param_branches: List[Dict[str, List[Dict]]],
                 param_consensus: Dict,
                 loss=keras.losses.CategoricalCrossentropy(),
                 optimizer=keras.optimizers.Adam(),
                 metrics=['accuracy'],
                 compile: Dict = None):
        super().__init__()

        if len(param_branches) != number_branches:
            ValueError('Number of branches and number of parameter sets different')
        if compile is None:
            compile = {}

        branched_models = []
        self.number_branches = number_branches

        input_lay = keras.Input(shape=input_size)

        for i in range(number_branches):
            parameters = param_branches[i]
            utils.check_params(parameters)
            index = {}
            arch = []

            for key in parameters.keys():
                index[key] = 0

            architecture = list(parameters.get('architecture'))
            for k, j in enumerate(architecture):
                if k == 0:
                    arch, index = utils.add_layer(j, input_lay, index, parameters, arch)
                else:
                    arch, index = utils.add_layer(j, arch[len(arch) - 1], index, parameters, arch)

            branched_models.append(arch[len(arch) - 1])

        x = Concatenate(axis=1)(branched_models)
        x = Dense(**param_consensus)(x)
        out = Dense(units=9, activation='softmax')(x)
        self.model = keras.Model(inputs=input_lay, outputs=out)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics, **compile)

    def fit(self, train_data, params_dataLoader: Dict = None, params_train: Dict = None):
        return super().fit(train_data, params_dataLoader, params_train)

    def evaluate(self, eval_data, params_dataLoader: Dict = None, params_eval: Dict = None):
        return super().evaluate(eval_data, params_dataLoader, params_eval)

    def predict(self, pred_data, params_dataLoader: Dict = None, params_predict: Dict = None):
        return super().predict(pred_data, params_dataLoader, params_predict)

    def summary(self):
        return super().summary()

    def save_model(self, path):
        super().save_model(path)

    def fit_and_evaluate(self, train_data, eval_data, callback: List[keras.callbacks.Callback] = None,
                         params_train_dataLoader: Dict = None,
                         params_eval_dataLoader: Dict = None,
                         params_train: Dict = None):
        return super().fit_and_evaluate(train_data, eval_data, callback,
                                        params_train_dataLoader,
                                        params_eval_dataLoader,
                                        params_train)

    def print_model(self, path):
        super().print_model(path)