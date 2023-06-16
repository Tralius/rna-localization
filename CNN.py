from typing import Dict
from model import Model
import keras
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout
import pandas as pd
from GeneDataLoader import GeneDataLoader

class CNN(Model):
    def __init__(self, 
                 architecure: str,
                 optimizer: str = 'adam',
                 loss: str = 'categorical_crossentropy',
                 metrics: list[str] = ['accuracy'],
                 dropouts: list = [],
                 conv: list[Dict] = [],
                 pooling: list = [], 
                 dense: list[Dict] = [],
                 **kwargs) -> None:

        arch = list(architecure)
        if len(dropouts)!=arch.count('d'):
            ValueError('number of dropouts not equal to number of dropout parameters')
        if len(conv) != arch.count('c'):
            ValueError('number of convolutional 1D layers not equal to number of convolutional parameters')
        if len(pooling) != arch.count('p'):
            ValueError('number of max pooling layers not equal to number of pooling parameters')
        if len(dense) != arch.count('e'):
            ValueError('number of dense layers not equal to number of dense parameters')

        self.model = keras.Sequential()

        for i in arch:
            if i=='c':
                self.model.add(Conv1D(**conv.pop(0)))
            if i=='d':
                self.model.add(Dropout(dropouts.pop(0)))
            if i=='e':
                self.model.add(Dense(**dense.pop(0)))
            if i=='f':
                self.model.add(Flatten())
            if i=='p':
                self.model.add(MaxPooling1D(pooling.pop(0)))

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    
    def fit(self, train_data: pd.DataFrame, **kwargs):
        padding_length = kwargs.pop('padding_length')
        batch_size_train = kwargs.pop('batch_size_train')
        shuffle_batch_train = kwargs.pop('shuffle_batch_train')
        train_data_loader = GeneDataLoader(train_data, padding_length=padding_length, batch_size=batch_size_train, shuffle=shuffle_batch_train)
        return self.model.fit(x=train_data_loader, **kwargs)
    
    def evaluate(self, eval_data: pd.DataFrame, **kwargs):
        padding_length = kwargs.pop('padding_length')
        batch_size_valid = kwargs.pop('batch_size_valid')
        shuffle_batch_valid = kwargs.pop('shuffle_batch_valid')
        validation_data_loader = GeneDataLoader(eval_data, padding_length=padding_length, batch_size=batch_size_valid, shuffle=shuffle_batch_valid)
        return self.model.evaluate(x=validation_data_loader, **kwargs)
