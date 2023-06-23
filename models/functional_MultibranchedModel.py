import keras
import numpy as np
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, MultiHeadAttention, Reshape, LeakyReLU, \
    BatchNormalization, Concatenate
from keras.utils import plot_model
from notes.model import Model
from typing import Dict, Tuple
from collections import Counter
from dataloaders.GeneDataLoader import GeneDataLoader


class MultiBranch(Model):
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
                 param_branches: list[Dict],
                 param_consensus: Dict,
                 loss = keras.losses.CategoricalCrossentropy(),
                 optimizer = keras.optimizers.Adam(),
                 metrics = ['accuracy'],
                 plot: bool = False,
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
            check_params(parameters)
            index = {}
            for key in parameters.keys():
                index[key] = 0

            architecture = list(parameters.get('architecture'))
            for k, j in enumerate(architecture):
                if k == 0:
                    if j == 'a':
                        x = MultiHeadAttention(**parameters.get('attention')[index.get('attention')])(input_lay)
                        index['attention'] = index.get('attention') + 1
                    elif j == 'b':
                        x = BatchNormalization()(input_lay)
                        index['batch'] = index.get('batch') + 1
                    elif j == 'c':
                        x = Conv1D(**parameters.get('conv')[index.get('conv')])(input_lay)
                        index['conv'] = index.get('conv') + 1
                    elif j == 'd':
                        x = Dropout(**parameters.get('dropouts')[index.get('dropouts')])(input_lay)
                        index['dropouts'] = index.get('dropouts') + 1
                    elif j == 'e':
                        x = Dense(**parameters.get('dense')[index.get('dense')])(input_lay)
                        index['dense'] = index.get('dense') + 1
                    elif j == 'f':
                        x = Flatten()(input_lay)
                    elif j == 'l':
                        x = LeakyReLU(**parameters.get('leaky')[index.get('leaky')])(input_lay)
                        index['leaky'] = index.get('leaky') + 1
                    elif j == 'p':
                        x = MaxPooling1D(**parameters.get('pooling')[index.get('pooling')])(input_lay)
                        index['pooling'] = index.get('pooling') + 1
                    elif j == 'r':
                        x = Reshape(**parameters.get('reshape')[index.get('reshape')])(input_lay)
                        index['reshape'] = index.get('reshape') + 1
                else:
                    if j == 'a':
                        x = MultiHeadAttention(**parameters.get('attention')[index.get('attention')])(x)
                        index['attention'] = index.get('attention') + 1
                    elif j == 'b':
                        x = BatchNormalization()(x)
                        index['batch'] = index.get('batch') + 1
                    elif j == 'c':
                        x = Conv1D(**parameters.get('conv')[index.get('conv')])(x)
                        index['conv'] = index.get('conv') + 1
                    elif j == 'd':
                        x = Dropout(**parameters.get('dropouts')[index.get('dropouts')])(x)
                        index['dropouts'] = index.get('dropouts') + 1
                    elif j == 'e':
                        x = Dense(**parameters.get('dense')[index.get('dense')])(x)
                        index['dense'] = index.get('dense') + 1
                    elif j == 'f':
                        x = Flatten()(x)
                    elif j == 'l':
                        x = LeakyReLU(**parameters.get('leaky')[index.get('leaky')])(x)
                        index['leaky'] = index.get('leaky') + 1
                    elif j == 'p':
                        x = MaxPooling1D(**parameters.get('pooling')[index.get('pooling')])(x)
                        index['pooling'] = index.get('pooling') + 1
                    elif j == 'r':
                        x = Reshape(**parameters.get('reshape')[index.get('reshape')])(x)
                        index['reshape'] = index.get('reshape') + 1

            branched_models.append(x)

        x = Concatenate(axis=1)(branched_models)
        out = Dense(units=9, **param_consensus)(x)
        self.model = keras.Model(inputs=input_lay, outputs=out)
        if plot:
            plot_model(self.model, show_shapes=True)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics, **compile)

    def fit(self, train_data, params_dataLoader: Dict = None, params_train: Dict = None):
        if params_train is None:
            Warning('evalutation with default parameters')
            params_train = {}
        if params_dataLoader is None:
            Warning('data Loader uses default arguments')
            params_dataLoader = {}
        return self.model.fit(GeneDataLoader(train_data, **params_dataLoader), **params_train)

    def evaluate(self, eval_data, params_eval: Dict = None, params_dataLoader: Dict = None):
        if params_eval is None:
            Warning('evalutation with default parameters')
            params_eval = {}
        if params_dataLoader is None:
            Warning('data Loader uses default arguments')
            params_dataLoader = {}
        return self.model.evaluate(GeneDataLoader(eval_data, **params_dataLoader), **params_eval)

    def predict(self, data, params_dataLoader: Dict = None, params_predict: Dict = None):
        if params_predict is None:
            Warning('evalutation with default parameters')
            params_predict = {}
        if params_dataLoader is None:
            Warning('data Loader uses default arguments')
            params_dataLoader = {}
        return self.model.predict(GeneDataLoader(data, **params_dataLoader), **params_predict)


def check_params(parameters: Dict):
    architecture = parameters.get('architecture')
    if architecture is None:
        ValueError('No architecture given')

    architecture = list(architecture)

    occurences = Counter(architecture)

    for layer, occ in occurences.items():
        if layer == 'd':
            dropouts = parameters.get('dropouts')
            if occ != len(dropouts):
                ValueError('number of dropouts not equal to number of dropout parameters')
        elif layer == 'c':
            conv = parameters.get('conv')
            if occ != len(conv):
                ValueError('number of convolutional 1D layers not equal to number of convolutional parameters')
        elif layer == 'p':
            padding = parameters.get('pooling')
            if occ != len(padding):
                ValueError('number of max pooling layers not equal to number of pooling parameters')
        elif layer == 'e':
            dense = parameters.get('dense')
            if occ != len(dense):
                ValueError('number of dense layers not equal to number of dense parameters')
        elif layer == 'a':
            attention = parameters.get('attention')
            if occ != len(attention):
                ValueError('number of multihead attention not equal to number of attention parameters')
        elif layer == 'r':
            reshape = parameters.get('reshape')
            if occ != len(reshape):
                ValueError('number of reshape layer not equal to number of reshape parameters')
        elif layer == 'l':
            reshape = parameters.get('leaky')
            if occ != len(reshape):
                ValueError('number of leakyReLU layer not equal to number of leaky parameters')
        elif layer == 'b':
            reshape = parameters.get('batch')
            if occ != len(reshape):
                ValueError('number of batch normalization layers not equal to number of batch parameters')
        else:
            NotImplementedError()