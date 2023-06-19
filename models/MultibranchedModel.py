import keras
import numpy as np
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, MultiHeadAttention
from keras.losses import MeanSquaredError
from model import Model
from typing import Dict, Tuple
from collections import Counter
from dataloaders.GeneDataLoader import GeneDataLoader
import pandas as pd


class MultiBranchMultiHead(Model):
    """

    TODO: refine docs
    ...

    Architecture:
    d : dropouts
    c

    """

    def __init__(self,
                 param_branches: list[Dict],
                 param_consensus: Dict,
                 number_branches: int = 3,
                 training: list[Dict] = None,
                 training_consensus: Dict = None,
                 **kwargs):
        super().__init__(**kwargs)
        if len(param_branches) != number_branches:
            ValueError('Number of branches and number of parameter sets different')

        if training is None:
            Warning('Training all models with default variables')
        elif len(training) < number_branches:
            Warning(
                f'Number of branches greater than provided training parameters. Last {number_branches - len(training)} '
                f'models will be trained with default variables.')
        if len(training) > number_branches:
            Warning(f'Number of branches less than provided training parameters. Only the first {number_branches}'
                    f' parameters will be used.')

        self.branched_models = []
        self.number_branches = number_branches

        for i in range(number_branches):
            self.branched_models[i] = keras.Sequential()

            parameters = param_branches[i]
            check_params(parameters)
            index = {}
            for key in parameters.keys():
                index[key] = 0

            architecture = list(parameters.get('architecture'))
            for j in architecture:
                if j == 'd':
                    self.branched_models[i].add(Dropout(**parameters.get('dropouts')[index.get('d')]))
                    index['d'] = index.get('d') + 1
                elif j == 'c':
                    self.branched_models[i].add(Conv1D(**parameters.get('conv')[index.get('c')]))
                    index['c'] = index.get('c') + 1
                elif j == 'p':
                    self.branched_models[i].add(MaxPooling1D(**parameters.get('pooling')[index.get('p')]))
                    index['p'] = index.get('p') + 1
                elif j == 'e':
                    self.branched_models[i].add(Dense(**parameters.get('dense')[index.get('e')]))
                    index['e'] = index.get('e') + 1
                elif j == 'a':
                    self.branched_models[i].add(MultiHeadAttention(**parameters.get('attention')[index.get('a')]))
                    index['a'] = index.get('a') + 1

            if i >= len(training):
                self.branched_models[i].compile()
            else:
                self.branched_models[i].compile(**training[i])

        self.final_merge_model = keras.Sequential()
        self.final_merge_model.add(Dense(units=9, **param_consensus))
        self.final_merge_model.compile(loss=MeanSquaredError, **training_consensus)

    def fit(self, train_data, params_branched: list[Dict] = None, params_consensus: Dict = None,
            params_loader: Dict = None):
        if params_branched is None:
            Warning('Training all models with default variables')
        elif len(params_branched) < self.number_branches:
            Warning(
                f'Number of branches greater than provided training parameters. Last {self.number_branches - len(params_branched)} '
                f'models will be trained with default variables.')
        if len(params_branched) > self.number_branches:
            Warning(f'Number of branches less than provided training parameters. Only the first {self.number_branches}'
                    f' parameters will be used.')

        dataLoader = GeneDataLoader(train_data, **params_loader)
        branches_pred_x = []
        branches_pred_y = []

        for x_train, y_train in dataLoader:
            for i, model in enumerate(self.branched_models):
                if i >= len(params_branched):
                    model.fit(x_train, y_train)
                else:
                    model.fit(x_train, y_train, **params_branched[i])

            results_branched = (self.branched_models[i].predict(x_train) for i in range(self.number_branches))
            branches_pred_x.append(np.concatenate(results_branched, axis=1))
            branches_pred_y.append(y_train)

        pred_x_concat = np.concatenate(branches_pred_x, axis=0)
        pred_y_concat = np.concatenate(branches_pred_y, axis=0)
        return self.final_merge_model.fit(pred_x_concat, pred_y_concat, **params_consensus)


    def evaluate(self, eval_data, params_branched: list[Dict] = None, params_consensus: Dict = None,
                 params_loader: Dict = None):
        if params_branched is None:
            Warning('Evaluate all models with default variables')
        elif len(params_branched) < self.number_branches:
            Warning(
                f'Number of branches greater than provided evaluation parameters. Last {self.number_branches - len(params_branched)} '
                f'models will be trained with default variables.')
        if len(params_branched) > self.number_branches:
            Warning(
                f'Number of branches less than provided evaluation parameters. Only the first {self.number_branches}'
                f' parameters will be used.')

        dataLoader = GeneDataLoader(eval_data, **params_loader)
        pred_x_concat, pred_y_concat = self.predict_branches(dataLoader)

        return self.final_merge_model.evaluate(pred_x_concat, pred_y_concat, **params_consensus)

    def predict(self, data, params_loader: Dict = None, params_predict: Dict = None, **kwargs):
        dataLoader = GeneDataLoader(data, **params_loader)
        pred_x_concat, _ = self.predict_branches(dataLoader)

        return self.final_merge_model.predict(pred_x_concat, **params_predict)

    def predict_branches(self, dataLoader: keras.Sequential) -> Tuple[np.ndarray, np.ndarray]:
        branches_pred_x = []
        branches_pred_y = []

        for x_eval, y_eval in dataLoader:
            results_branched = (self.branched_models[i].predict(x_eval) for i in range(self.number_branches))
            branches_pred_x.append(np.concatenate(results_branched, axis=1))
            branches_pred_y.append(y_eval)

        pred_x_concat = np.concatenate(branches_pred_x, axis=0)
        pred_y_concat = np.concatenate(branches_pred_y, axis=0)

        return pred_x_concat, pred_y_concat



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
                ValueError('number of multihead attention not equal to number of dense parameters')
        else:
            NotImplementedError()
