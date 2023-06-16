from typing import Dict
from model import Model
import keras
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout

class CNN(Model):
    def __init__(self, 
                 train_data,
                 valid_data,
                 architecure: str, 
                 optimizer: str = 'adam',
                 loss: str = 'categorical_crossentropy',
                 metrics: list[str] = ['accuracy'],
                 dropouts: list = [],
                 conv: list[Dict] = [],
                 pooling: list = [], 
                 dense: list[Dict] = [],
                 **kwargs) -> None:
        super().__init__(train_data=train_data, validation_data=valid_data, **kwargs)

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

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    
    def fit(self, **kwargs):
        return self.model.fit(x=self.train_data_loader, **kwargs)
    
    def evaluate(self, **kwargs):
        return self.model.evaluate(x=self.validation_data_loader, **kwargs)
