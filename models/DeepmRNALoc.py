from tensorflow import keras
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.losses import MeanSquaredError
import tensorflow_probability as tfp
from keras.metrics import Metric
import tensorflow as tf
# DeepmRNALoc model from https://github.com/Thales-research-institute/DeepmRNALoc

class Pearson(Metric):

    def __init__(self, name='person', sample_axis=0, event_axis=None, keepdims=False, eps=1e-3,
                 return_dict: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        self.sample_axis = sample_axis
        self.event_axis = event_axis
        self.keepdims = keepdims
        self.eps = eps
        self.return_dict = return_dict
        self.corr = None

    def update_state(self, y_true, y_pred):
        y_true_std = tfp.stats.stddev(y_true, sample_axis=self.sample_axis, keepdims=True)
        y_pred_std = tfp.stats.stddev(y_pred, sample_axis=self.sample_axis, keepdims=True)
        if y_true_std == 0:
            y_true /= self.eps
        else:
            y_true /= (y_true + np.sign(y_true_std) * self.eps)
        if y_pred is not None:
            if y_pred_std == 0:
                y_pred /= self.eps
            else:
                y_pred /= (y_pred_std + np.sign(y_pred_std) * self.eps)

        result = tfp.stats.covariance(x=y_true,
                                      y=y_pred,
                                      event_axis=self.event_axis,
                                      sample_axis=self.sample_axis,
                                      keepdims=self.keepdims)

        if self.return_dict:
            res_dict = {}
            res_dict['ERM'] = result[0]
            res_dict['KDEL'] = result[1]
            res_dict['LMA'] = result[2]
            res_dict['MITO'] = result[3]
            res_dict['NES'] = result[4]
            res_dict['NIK'] = result[5]
            res_dict['NLS'] = result[6]
            res_dict['NUCP'] = result[7]
            res_dict['OMM'] = result[8]
            self.corr = res_dict
        else:
            self.corr = result

    def result(self):
        return self.corr

    def reset_states(self):
        self.corr = None



def pearson(y_true, y_pred, sample_axis=0,
            event_axis=None,
            keepdims=False,
            eps=0.001):
    y_true /= (tfp.stats.stddev(y_true, sample_axis=sample_axis, keepdims=True) + eps)
    if y_pred is not None:
        y_pred /= (tfp.stats.stddev(y_pred, sample_axis=sample_axis, keepdims=True) + eps)

    return tfp.stats.covariance(
        x=y_true,
        y=y_pred,
        event_axis=event_axis,
        sample_axis=sample_axis,
        keepdims=keepdims)

def build_model(max_len, layer_size: int = 128, learning_rate = 1e-3, dropout_rate = 0.2, out_size = 9):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(max_len, 4)))
    model.add(keras.layers.Reshape((max_len*4, 1)))
    model.add(keras.layers.Conv1D(64, 3,strides=2,padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Conv1D(64, 3,strides=1,padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Conv1D(128, 3,strides=2,padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Conv1D(128, 3,strides=1,padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Conv1D(256, 3,strides=2,padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Conv1D(256, 3,strides=1,padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Conv1D(512, 3,strides=2,padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Conv1D(512, 3,strides=1,padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Dropout(dropout_rate))
#LSTM
    model.add(Bidirectional(keras.layers.LSTM(512, return_sequences=True)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(Bidirectional(keras.layers.LSTM(512, return_sequences=False)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Dropout(dropout_rate))
#FCN
    model.add(keras.layers.Dense(layer_size,kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Dense(layer_size*2,kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Dense(layer_size*4,kernel_initializer='glorot_uniform'))
    model.add(keras.layers.Dense(9, activation="softmax")) # Changed output size to 9
    loss = CategoricalCrossentropy(label_smoothing=0.01)
    #loss = MeanSquaredError()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.9)
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), metrics=[keras.metrics.MeanSquaredError()])
    return model





def build_net(max_len, layer_size=128, learning_rate=1e-3, dropout_rate=0.2, out_size=9):
    model = keras.models.Sequential()
    #model.add(keras.layers.Flatten(input_shape=(max_len, 4)))
    #model.add(keras.layers.Reshape((max_len * 4, 1)))
    model.add(keras.layers.Conv1D(64, 12, strides=3, activation='relu', input_shape=(max_len, 4)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv1D(64, 3, strides=2, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Conv1D(64, 3, strides=1, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Conv1D(128, 3, strides=2, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Conv1D(128, 3, strides=1, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Conv1D(256, 3, strides=2, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Conv1D(256, 3, strides=1, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Conv1D(512, 3, strides=2, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Conv1D(512, 3, strides=1, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Dropout(dropout_rate))

    # FCN
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(layer_size, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Dense(layer_size * 2, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Dense(layer_size * 4, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Dense(layer_size * 8, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.Dense(9, activation="softmax"))  # Changed output size to 9
    loss = CategoricalCrossentropy()
    # loss = MeanSquaredError()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.1)
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.MeanSquaredError()])
    return model
def build_lstm(max_len, layer_size=128, learning_rate=1e-3, dropout_rate=0.2, out_size=9):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv1D(64, 12, strides=3, activation='relu', input_shape=(max_len, 4)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv1D(64, 3, strides=2, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Conv1D(64, 3, strides=1, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Conv1D(128, 3, strides=2, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Conv1D(128, 3, strides=1, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.LSTM(units=128))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(layer_size, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Dense(layer_size * 2, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Dense(layer_size * 4, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.Dense(9, activation="softmax"))  # Changed output size to 9

    loss = CategoricalCrossentropy()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.1)
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=[keras.metrics.MeanSquaredError(), pearson])
    return model







