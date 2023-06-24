from typing import Dict
from notes.model import Model
import keras
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, MultiHeadAttention
from keras.utils import plot_model
import pandas as pd
from dataloaders.GeneDataLoader import GeneDataLoader


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
                 architecure: str,
                 optimizer: str = 'adam',
                 loss: str = 'categorical_crossentropy',
                 metrics: list[str] = None,
                 dropouts: list[Dict] = None,
                 conv: list[Dict] = None,
                 pooling: list[Dict] = None,
                 dense: list[Dict] = None,
                 attention: list[Dict] = None,
                 epochs: int = 5,
                 **kwargs) -> None:

        super().__init__()

        if metrics is None: metrics = ['accuracy']
        if dropouts is None: dropouts = []
        if conv is None: conv = []
        if pooling is None: pooling = []
        if dense is None: dense = []
        if attention is None: attention = []

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
                index['c'] = index.get('c') + 1
            if i == 'd':
                self.model.add(Dropout(dropouts[index.get('d')]))
                index['d'] = index.get('d') + 1
            if i == 'e':
                self.model.add(Dense(**dense[index.get('e')]))
                index['e'] = index.get('e') + 1
            if i == 'f':
                self.model.add(Flatten())
            if i == 'p':
                self.model.add(MaxPooling1D(pooling[index.get('p')]))
                index['p'] = index.get('p') + 1
            if i == 'a':
                self.model.add(MultiHeadAttention(attention[index.get('a')]))
                index['a'] = index.get('a') + 1

        self.epochs = epochs

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def fit(self, train_data: pd.DataFrame, params_loader: Dict = None, **kwargs):
        padding_length = kwargs.pop('padding_length')
        batch_size_train = kwargs.pop('batch_size_train')
        shuffle_batch_train = kwargs.pop('shuffle_batch_train')
        train_data_loader = GeneDataLoader(train_data, padding_length=padding_length, batch_size=batch_size_train,
                                           shuffle=shuffle_batch_train)
        return self.model.fit(train_data_loader, epochs=self.epochs, **kwargs)

    def evaluate(self, eval_data: pd.DataFrame, params_loader: Dict = None, **kwargs):
        padding_length = kwargs.pop('padding_length')
        batch_size_valid = kwargs.pop('batch_size_valid')
        shuffle_batch_valid = kwargs.pop('shuffle_batch_valid')
        validation_data_loader = GeneDataLoader(eval_data, padding_length=padding_length, batch_size=batch_size_valid,
                                                shuffle=shuffle_batch_valid)
        return self.model.evaluate(validation_data_loader, **kwargs)

    def fit_and_evaluate(self, train_data, eval_data, **kwargs):
        padding_length = kwargs.pop('padding_length')
        batch_size = kwargs.pop('batch_size_train')
        shuffle_batch = kwargs.pop('shuffle_batch_train')
        callbacks = kwargs.pop('callbacks')
        train_data_loader = GeneDataLoader(train_data, padding_length=padding_length, batch_size=batch_size,
                                           shuffle=True)
        validation_data_loader = GeneDataLoader(eval_data, padding_length=padding_length, batch_size=batch_size,
                                                shuffle=False)
        return self.model.fit(train_data_loader, epochs=self.epochs, callbacks=callbacks, validation_data=validation_data_loader, **kwargs)

    def predict(self, data, params_loader: Dict = None, params_predict: Dict = None):
        dataLoader = GeneDataLoader(data, **params_loader)
        return self.model.predict(dataLoader, **params_predict)

    def print_model(self, path):
        self.model.summary()
        plot_model(self.model, path, show_shapes=True)

    def save_model(self, path):
        self.model.save(path)
