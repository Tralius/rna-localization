from tensorflow import keras
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.losses import MeanSquaredError

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





def build_lstm(max_len, layer_size=128, learning_rate=1e-3, dropout_rate=0.03, out_size=9):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(max_len, 4)))
    model.add(keras.layers.Reshape((max_len * 4, 1)))
    model.add(keras.layers.Conv1D(64, 3, strides=3, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv1D(64, 3, strides=2, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Conv1D(64, 3, strides=1, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Dropout(dropout_rate))sla
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
    loss = CategoricalCrossentropy(label_smoothing=0.01)
    # loss = MeanSquaredError()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.9)
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=[keras.metrics.MeanSquaredError()])
    return model
