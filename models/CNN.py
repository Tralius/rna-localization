from typing import Dict
from model import Model
import keras
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, MultiHeadAttention
import pandas as pd
from dataloaders.GeneDataLoader import GeneDataLoader


class CNN(Model):
    def __init__(self,
                 architecure: str,
                 optimizer: str = 'adam',
                 loss: str = 'categorical_crossentropy',
                 metrics: list[str] = ['accuracy'],
                 dropouts: list[Dict] = [],
                 conv: list[Dict] = [],
                 pooling: list[Dict] = [],
                 dense: list[Dict] = [],
                 attention: list[Dict] = [],
                 **kwargs) -> None:

        super().__init__()
        arch = list(architecure)
        if len(dropouts) != arch.count('d'):
            ValueError('number of dropouts not equal to number of dropout parameters')
        if len(conv) != arch.count('c'):
            ValueError('number of convolutional 1D layers not equal to number of convolutional parameters')
        if len(pooling) != arch.count('p'):
            ValueError('number of max pooling layers not equal to number of pooling parameters')
        if len(dense) != arch.count('e'):
            ValueError('number of dense layers not equal to number of dense parameters')
        if len(attention) != arch.count('a'):
            ValueError('number of attention layers not equal to number of attention parameters')

        index = {'a': 0, 'c': 0, 'd': 0, 'e': 0, 'p': 0}

        self.model = keras.Sequential()

        for i in arch:
            if i == 'c':
                self.model.add(Conv1D(**conv[index.get('c')]))
            if i == 'd':
                self.model.add(Dropout(dropouts[index.get('d')]))
            if i == 'e':
                self.model.add(Dense(**dense[index.get('e')]))
            if i == 'f':
                self.model.add(Flatten())
            if i == 'p':
                self.model.add(MaxPooling1D(pooling[index.get('p')]))
            if i == 'm':
                self.model.add(MultiHeadAttention(attention[index.get('a')]))

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def fit(self, train_data: pd.DataFrame, params_loader: Dict = None, **kwargs):
        padding_length = kwargs.pop('padding_length')
        batch_size_train = kwargs.pop('batch_size_train')
        shuffle_batch_train = kwargs.pop('shuffle_batch_train')
        train_data_loader = GeneDataLoader(train_data, padding_length=padding_length, batch_size=batch_size_train,
                                           shuffle=shuffle_batch_train)
        return self.model.fit(x=train_data_loader, **kwargs)

    def evaluate(self, eval_data: pd.DataFrame, params_loader: Dict = None, **kwargs):
        padding_length = kwargs.pop('padding_length')
        batch_size_valid = kwargs.pop('batch_size_valid')
        shuffle_batch_valid = kwargs.pop('shuffle_batch_valid')
        validation_data_loader = GeneDataLoader(eval_data, padding_length=padding_length, batch_size=batch_size_valid,
                                                shuffle=shuffle_batch_valid)
        return self.model.evaluate(x=validation_data_loader, **kwargs)

    def predict(self, data, params_loader: Dict = None, params_predict: Dict = None):
        dataLoader = GeneDataLoader(data, **params_loader)
        return self.model.predict(dataLoader, **params_predict)
