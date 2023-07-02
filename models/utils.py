from typing import Dict, List
from collections import Counter
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, Reshape, LeakyReLU, \
    BatchNormalization, add, Layer, Activation, Lambda, Multiply
from keras import backend as K

def check_params(parameters: Dict):
    architecture = parameters.get('architecture')
    if architecture is None:
        ValueError('No architecture given')

    occurences = Counter(list(architecture))

    for layer, occ in occurences.items():
        if layer == 'a':
            attention = parameters.get('attention')
            if occ != len(attention):
                ValueError('number of multihead attention not equal to number of attention parameters')
        elif layer == 'b':
            reshape = parameters.get('batch')
            if occ != len(reshape):
                ValueError('number of batch normalization layers not equal to number of batch parameters')
        elif layer == 'c':
            conv = parameters.get('conv')
            if occ != len(conv):
                ValueError('number of convolutional 1D layers not equal to number of convolutional parameters')
        elif layer == 'd':
            dropouts = parameters.get('dropouts')
            if occ != len(dropouts):
                ValueError('number of dropouts not equal to number of dropout parameters')
        elif layer == 'e':
            dense = parameters.get('dense')
            if occ != len(dense):
                ValueError('number of dense layers not equal to number of dense parameters')
        elif layer == 'l':
            leaky = parameters.get('leaky')
            if occ != len(leaky):
                ValueError('number of leakyReLU layer not equal to number of leaky parameters')
        elif layer == 'p':
            padding = parameters.get('pooling')
            if occ != len(padding):
                ValueError('number of max pooling layers not equal to number of pooling parameters')
        elif layer == 'r':
            reshape = parameters.get('reshape')
            if occ != len(reshape):
                ValueError('number of reshape layer not equal to number of reshape parameters')
        elif layer == 's':
            skip = parameters.get('skip')
            if occ != len(skip):
                ValueError('number of skip connection layer not equal to number of skip parameters')
        else:
            NotImplementedError()

def add_layer(layer: str, arg, index: Dict, params: Dict, arch: List):
    if layer == 'a':
        arch.append(Attention(**params.get('attention')[index.get('attention')])(arg))
        index['attention'] = index.get('attention') + 1
        return arch, index
    elif layer == 'b':
        arch.append(BatchNormalization()(arg))
        index['batch'] = index.get('batch') + 1
        return arch, index
    elif layer == 'c':
        arch.append(Conv1D(**params.get('conv')[index.get('conv')])(arg))
        index['conv'] = index.get('conv') + 1
        return arch, index
    elif layer == 'd':
        arch.append(Dropout(**params.get('dropouts')[index.get('dropouts')])(arg))
        index['dropouts'] = index.get('dropouts') + 1
        return arch, index
    elif layer == 'e':
        arch.append(Dense(**params.get('dense')[index.get('dense')])(arg))
        index['dense'] = index.get('dense') + 1
        return arch, index
    elif layer == 'f':
        arch.append(Flatten()(arg))
        return arch, index
    elif layer == 'l':
        arch.append(LeakyReLU(**params.get('leaky')[index.get('leaky')])(arg))
        index['leaky'] = index.get('leaky') + 1
        return arch, index
    elif layer == 'p':
        arch.append(MaxPooling1D(**params.get('pooling')[index.get('pooling')])(arg))
        index['pooling'] = index.get('pooling') + 1
        return arch, index
    elif layer == 'r':
        arch.append(Reshape(**params.get('reshape')[index.get('reshape')])(arg))
        index['reshape'] = index.get('reshape') + 1
        return arch, index
    elif layer == 's':
        adding_index = params.get('skip')[index.get('skip')]
        adding = [arch[i] for i in adding_index].get('index').append(arg)
        arch.append(add()(adding))
        index['skip'] = index.get('skip') + 1
        return arch, index

class Attention(Layer):
    def __init__(self, attention_size, activation_dense='tanh', activation_act='softmax', **kwargs):
        super().__init__()
        self.dense1 = Dense(units=attention_size, activation=activation_dense)
        self.dense2 = Dense(units=1, use_bias=False)
        self.activation = Activation(activation=activation_act)
        self.lam = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))

    def call(self, inputs):
        context = self.dense1(inputs)
        attention = self.dense2(context)
        scores = Flatten()(attention)
        attention_weights = Reshape(target_shape=(2146, 1))(scores)
        output = self.lam(Multiply()([inputs, attention_weights]))
        return output
