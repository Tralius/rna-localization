import keras
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, MultiHeadAttention
from keras.losses import MeanSquaredError
from model import Model
from typing import Dict, Union
from collections import Counter
from GeneDataLoader import GeneDataLoader
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
                elif j == 'm':
                    self.branched_models[i].add(MultiHeadAttention(**parameters.get('multihead')[index.get('m')]))
                    index['m'] = index.get('m') + 1

            if i >= len(training):
                self.branched_models[i].compile()
            else:
                self.branched_models[i].compile(**training[i])

        self.model = keras.Sequential()  # TODO: remane in "final_merge_model"
        self.model.add(Dense(units=9, **param_consensus))
        self.model.compile(loss=MeanSquaredError, **training_consensus)

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
        for x_train, y_train in dataLoader:
            for i, model in enumerate(self.branched_models):
                if i >= len(params_branched):
                    model.fit(x_train, y_train)
                else:
                    model.fit(x_train, y_train, **params_branched[i])

            results_branched = [self.branched_models[i].evaluate() for i in # TODO: no eval
                                range(self.number_branches)]  # TODO right command for getting results only
            branches_pred = pd.concat(results_branched, axis=0)  # TODO adjust to numpy

            self.model.fit(branches_pred, y_train, **params_consensus)

    def evaluate(self, eval_data, params_branched: list[Dict] = None, params_consensus: Dict = None,
                 params_loader: Dict = None):
        if params_branched is None:
            Warning('Evaluate all models with default variables')
        elif len(params_branched) < self.number_branches:
            Warning(
                f'Number of branches greater than provided evaluation parameters. Last {self.number_branches - len(params_branched)} '
                f'models will be trained with default variables.')
        if len(params_branched) > self.number_branches:
            Warning(f'Number of branches less than provided evaluation parameters. Only the first {self.number_branches}'
                    f' parameters will be used.')

        dataLoader = GeneDataLoader(eval_data, **params_loader)
        for x_eval, y_eval in dataLoader:
            for i, model in enumerate(self.branched_models):



def check_params(parameters: Dict):
    architecture = parameters.get('architecture')
    if architecture is None:
        ValueError('No architecture given')

    architecture = list(architecture)

    parts_of_architecture = Counter(architecture).keys()
    occurences = Counter(architecture).values()

    for i in range(len(parts_of_architecture)):
        if parts_of_architecture[i] == 'd':
            dropouts = parameters.get('dropouts')
            if occurences[i] != len(dropouts):
                ValueError('number of dropouts not equal to number of dropout parameters')
        elif parts_of_architecture[i] == 'c':
            conv = parameters.get('conv')
            if occurences[i] != len(conv):
                ValueError('number of convolutional 1D layers not equal to number of convolutional parameters')
        elif parts_of_architecture[i] == 'p':
            padding = parameters.get('pooling')
            if occurences[i] != len(padding):
                ValueError('number of max pooling layers not equal to number of pooling parameters')
        elif parts_of_architecture[i] == 'e':
            dense = parameters.get('dense')
            if occurences[i] != len(dense):
                ValueError('number of dense layers not equal to number of dense parameters')
        elif parts_of_architecture[i] == 'm':
            multihead = parameters.get('multihead')
            if occurences[i] != len(multihead):
                ValueError('number of multihead attention not equal to number of dense parameters')
        else:
            NotImplementedError
