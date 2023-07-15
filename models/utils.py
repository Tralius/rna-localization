from typing import Dict, List
from collections import Counter
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, Reshape, LeakyReLU, \
    BatchNormalization, add, ReLU, GlobalAvgPool1D, Activation, Lambda, Multiply, Layer, Concatenate, 
from keras import backend as K
from keras.optimizers import SGD, Adam, Nadam
from keras.activations import tanh
import keras
from keras import regularizers



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
        elif layer == 'g':
            globalavg = parameters.get('globalavg')
            if occ != len(globalavg):
                ValueError('number of Global Average layers not equal to number of Glb Avg parameters')
        elif layer == 'e':
            dense = parameters.get('dense')
            if occ != len(dense):
                ValueError('number of dense layers not equal to number of dense parameters')
        elif layer == 'z':
            activation = parameters.get("activation")
            if occ != len(activation):
                ValueError('number of activation layer not equal to number of activation parameters')
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


def add_layer(layer: str, args, index: Dict, params: Dict, arch: List):
    if layer == 'a':
        result = [Attention(**params.get('attention')[index.get('attention')])(args[0])] + args[1:]
        arch.append(result)
        index['attention'] = index.get('attention') + 1
        return arch, index
    elif layer == 'b':
        result = [BatchNormalization()(args[0])] + args[1:]
        arch.append(result)
        index['batch'] = index.get('batch') + 1
        return arch, index
    elif layer == 'z':
        parameters = params.get('activation')[index.get('activation')]
        activation = None
        if parameters["type"] == "LeakyRelu":
            activation = LeakyReLU(float(parameters["alpha"]))
        elif parameters["type"] == "Relu":
            activation = ReLU()
        result = [activation(args[0])] + args[1:]
        arch.append(result)
        index['activation'] = index.get('activation') + 1
        return arch, index
    elif layer == 'c':
        result = [Conv1D(**params.get('conv')[index.get('conv')])(args[0])] + args[1:]
        arch.append(result)
        index['conv'] = index.get('conv') + 1
        return arch, index
    elif layer == 'i':
        concatlayer = Concatenate()
        c = concatlayer([args[0],args[1]])
        result = [c] + args[1:]
        arch.append(result)
        return arch, index
    elif layer == 'd':

        result = [Dropout(**params.get('dropouts')[index.get('dropouts')])(args[0])] + args[1:]
        arch.append(result)
        index['dropouts'] = index.get('dropouts') + 1
        return arch, index
    elif layer == 'e':

        result = [Dense(**params.get('dense')[index.get('dense')])(args[0])] + args[1:]
        arch.append(result)
        index['dense'] = index.get('dense') + 1
        return arch, index
    elif layer == 'f':
        result = [Flatten()(args[0])] + args[1:]
        arch.append(result)
        return arch, index
    elif layer == 'p':
        result = [MaxPooling1D(**params.get('pooling')[index.get('pooling')])(args[0])] + args[1:]
        arch.append(result)
        index['pooling'] = index.get('pooling') + 1
        return arch, index
    elif layer == 'r':
        result = [Reshape(**params.get('reshape')[index.get('reshape')])(args[0])] + args[1:]
        arch.append(result)
        index['reshape'] = index.get('reshape') + 1
        return arch, index
    elif layer == 'g':
        parameters = params.get('globalavg')
        parameters = parameters[index.get('globalavg')]
        result = None
        if parameters is None:
            result = GlobalAvgPool1D()(args[0])
        else:
            result = GlobalAvgPool1D(**parameters)(args[0])
        result = [result] + args[1:]
        arch.append(result)
        index['globalavg'] = index.get('globalavg') + 1
        return arch, index
    elif layer == 'l':
        result = [LSTM(**params.get('lstm')[index.get('lstm')])(args[0])] + args[1:]
        arch.append(result)
        index['lstm'] = index.get('lstm') + 1
        return arch, index
    elif layer == 's':
        parameters = params.get('skip')[index.get('skip')]
        # adding = [arch[i] for i in adding_index].get('index').append(arg)
        reg = None
        try:
            reg = parameters["kernel_regularizer"]
        except Exception as e:
            pass
        out = resblock(args[0], **parameters)
        arch.append([out] + args[1:])
        index['skip'] = index.get('skip') + 1
        return arch, index

def resblock(x, kernel_size, filters, use_bn, kernel_regularizer = None, padding = 'same', activation='relu', specific_reg = None, specific_act=None, use_bias=True):
    padding = 'same' # overwrite for now TODO!!
    if x.shape[-1] != filters:
        x = Conv1D(kernel_size= 1, filters= filters, padding=padding, activation=activation, kernel_initializer='he_normal')(x) # 1x1 conv to adjust kernel size
    fx = x
    if specific_act:
        if specific_act["type"] == "LeakyRelu":
            activation = LeakyReLU(float(specific_act["alpha"]))
        elif specific_act["type"] == "tanh":
            activation = tanh()
        
    if specific_reg:
        kernel_reg = specific_reg["kernel_regularizer"]
        bias_reg = specific_reg["bias_regularizer"]
        activity_reg = specific_reg["activity_regularizer"]
        kernel_reg = regularizers.L1L2(l1=float(kernel_reg["l1"]), l2=float(kernel_reg["l2"]))
        bias_reg = regularizers.L2(float(bias_reg["l2"]))
        activity_reg = regularizers.L2(float(activity_reg["l2"]))
        fx = Conv1D(kernel_size=kernel_size, 
                    kernel_regularizer = kernel_reg, 
                    bias_regularizer = bias_reg,
                    use_bias = use_bias,
                    activity_regularizer = activity_reg,
                    filters=filters, activation=activation, 
                    padding=padding)(fx)
    else:
        fx = Conv1D(kernel_size=kernel_size, 
                    filters=filters, 
                    activation=activation, 
                    kernel_regularizer=kernel_regularizer,
                    padding=padding)(fx)

    if use_bn:
        fx = BatchNormalization(scale=False)(fx)
    
    if specific_reg:
        kernel_reg = specific_reg["kernel_regularizer"]
        bias_reg = specific_reg["bias_regularizer"]
        activity_reg = specific_reg["activity_regularizer"]
        kernel_reg = regularizers.L1L2(l1=float(kernel_reg["l1"]), l2=float(kernel_reg["l2"]))
        bias_reg = regularizers.L2(float(bias_reg["l2"]))
        activity_reg = regularizers.L2(float(activity_reg["l2"]))
        fx = Conv1D(kernel_size=kernel_size, 
                    kernel_regularizer = kernel_reg, 
                    bias_regularizer = bias_reg,
                    use_bias = use_bias,
                    activity_regularizer = activity_reg,
                    filters=filters, activation=activation, 
                    padding=padding)(fx)
    else:
        fx = Conv1D(filters=filters, 
                    kernel_size=kernel_size, 
                    padding=padding, 
                    kernel_regularizer=kernel_regularizer,
                    use_bias = use_bias,)(fx)

    out = add([x, fx])
    #if use_bn:
    #    out = BatchNormalization()(out)
    final_activation = ReLU()
    if specific_act:
        if specific_act["type"] == "LeakyRelu":
            final_activation = LeakyReLU(float(specific_act["alpha"]))
        elif specific_act["type"] == "tanh":
            final_activation = tanh()
    out = final_activation(out)

    return out


#@keras.saving.register_keras_serializable(package='Attention')
class Attention(Layer):
    def __init__(self, attention_size, reshape_size, activation_dense='tanh', **kwargs):
        super().__init__()
        self.attention_size = attention_size
        self.activation_dense = activation_dense
        self.reshape_size = reshape_size
        
    def build(self, input_shape):
        self.dense1 = Dense(units=self.attention_size, activation=self.activation_dense)
        self.dense2 = Dense(units=1, use_bias=False)
        self.reshape = Reshape(target_shape=(self.reshape_size, 1))
        self.lam = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))
        self.mult = Multiply()

    def call(self, inputs):
        context = self.dense1(inputs)
        attention = self.dense2(context)
        scores = Flatten()(attention)
        attention_weights = self.reshape(scores)
        output = self.lam(self.mult([inputs, attention_weights]))
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'attention_size': self.attention_size,
                'reshape_size': self.reshape_size,
                'activation_dense': self.activation_dense
                }
            )
        return config


def set_optimizer(optimizer: str, learning_rate: float):
    if optimizer == 'adam':
        if learning_rate is None:
            return Adam()
        else:
            return Adam(learning_rate=learning_rate)
    if optimizer == 'sgd':
        if learning_rate is None:
            return SGD()
        else:
            return SGD(learning_rate=learning_rate)
    if optimizer == 'nadam':
        if learning_rate is None:
            return Nadam()
        else:
            return Nadam(learning_rate=learning_rate)

def extractY(data):
    testY = data.iloc[:, 0:9]
    sum_vec = testY.sum(axis=1)
    testY = testY.divide(sum_vec, axis='index')
    return testY
