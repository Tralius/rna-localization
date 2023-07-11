from tensorflow import keras
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Bidirectional, Input
from metrics import Pearson
from keras.models import Model
# DeepmRNALoc model from https://github.com/Thales-research-institute/DeepmRNALoc



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
def build_lstm(max_len, layer_size=128, learning_rate=1e-3, dropout_rate=0.2, out_size=9, input_size = 4):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(64, 12, strides=3, activation='relu', input_shape=(max_len, input_size)))
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
                  metrics=[keras.metrics.MeanSquaredError(), Pearson(return_dict=True)])
    return model


def build_m6A_model(max_len, layer_size=128, learning_rate=1e-3, dropout_rate=0.2, out_size=9, input_size = 5):
    seq_input = Input((max_len, input_size))
    m6A_input = Input((3,))

    conv1 = keras.layers.Conv1D(64, 12, strides=3, activation='relu')(seq_input)
    batchnorm1 = keras.layers.BatchNormalization()(conv1)
    conv2 = keras.layers.Conv1D(64, 3, strides=2, padding="same")(batchnorm1)
    bn2 = keras.layers.BatchNormalization()(conv2)
    LReLu1 = keras.layers.LeakyReLU(alpha=0.05)(bn2)
    conv3 = keras.layers.Conv1D(64, 3, strides=1, padding="same")(LReLu1)
    bn3 = keras.layers.BatchNormalization()(conv3)
    LReLu2 = keras.layers.LeakyReLU(alpha=0.05)(bn3)
    mp1d1 = keras.layers.MaxPooling1D(2)(LReLu2)
    drop1 = keras.layers.Dropout(dropout_rate)(mp1d1)
    conv4 = keras.layers.Conv1D(128, 3, strides=2, padding="same")(drop1)
    bn4 = keras.layers.BatchNormalization()(conv4)
    LReLu3 = keras.layers.LeakyReLU(alpha=0.05)(bn4)
    conv5 = keras.layers.Conv1D(128, 3, strides=1, padding="same")(LReLu3)
    bn5 = keras.layers.BatchNormalization()(conv5)
    LReLu4 = keras.layers.LeakyReLU(alpha=0.05)(bn5)
    mp1d2 = keras.layers.MaxPooling1D(2)(LReLu4)
    drop2 = keras.layers.Dropout(dropout_rate)(mp1d2)

    lstm1 = keras.layers.LSTM(units=128)(drop2)
    bn6 = keras.layers.BatchNormalization()(lstm1)
    LReLu4 = keras.layers.LeakyReLU(alpha=0.05)(bn6)
    drop3 = keras.layers.Dropout(dropout_rate)(LReLu4)

    flat_layer = keras.layers.Flatten()(drop3)

    # Concatenate the convolutional features and the vector input
    concat_layer = keras.layers.Concatenate()([m6A_input, flat_layer])
    dense1 = keras.layers.Dense(layer_size, kernel_initializer='glorot_uniform')(concat_layer)
    bn7 = keras.layers.BatchNormalization()(dense1)
    LReLu5 = keras.layers.LeakyReLU(alpha=0.05)(bn7)
    dense2 = keras.layers.Dense(layer_size * 2, kernel_initializer='glorot_uniform')(LReLu5)
    bn8 = keras.layers.BatchNormalization()(dense2)
    LReLu6 = keras.layers.LeakyReLU(alpha=0.05)(bn8)
    dense3 = keras.layers.Dense(layer_size * 4, kernel_initializer='glorot_uniform')(LReLu6)
    output = keras.layers.Dense(9, activation="softmax")(dense3)  # Changed output size to 9

    model = Model(inputs=[seq_input, m6A_input], outputs=output)

    loss = CategoricalCrossentropy()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.1)
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=[keras.metrics.MeanSquaredError(), Pearson(return_dict=True)])
    return model







