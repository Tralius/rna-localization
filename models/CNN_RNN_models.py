from tensorflow import keras
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Bidirectional, Input
from metrics import Pearson
from keras.models import Model

def m6A_model(max_len, learning_rate=1e-3, dropout_rate=0.5, input_size = 5):

    seq_input = Input((max_len, input_size))
    m6A_input = Input((3,))

    c1 = keras.layers.Conv1D(16, 17, activation='relu')(seq_input)
    p1 = keras.layers.MaxPooling1D(4)(c1)
    d1 = keras.layers.Dropout(dropout_rate)(p1)
    c2 = keras.layers.Conv1D(32, 11, padding = 'valid', activation='relu')(d1)
    p2 = keras.layers.MaxPooling1D(4)(c2)
    d2 = keras.layers.Dropout(dropout_rate)(p2)
    c3 = keras.layers.Conv1D(64, 7, padding='valid', activation='relu')(d2)
    p3 = keras.layers.MaxPooling1D(4)(c3)
    d3 = keras.layers.Dropout(dropout_rate)(p3)

    lstm1 = keras.layers.LSTM(units=256)(d3)
    bn1 = keras.layers.BatchNormalization()(lstm1)
    lrelu = keras.layers.LeakyReLU(alpha=0.05)(bn1)
    d4 = keras.layers.Dropout(dropout_rate)(lrelu)

    #f = keras.layers.Flatten()(d4)

    concat = keras.layers.Concatenate()([m6A_input, d4])
    e1 = keras.layers.Dense(512, activation='relu')(concat)
    d5 = keras.layers.Dropout(dropout_rate)(e1)
    output = keras.layers.Dense(9, activation='softmax')(d5)

    model = Model(inputs=[seq_input, m6A_input], outputs=output)

    loss = CategoricalCrossentropy()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.1)
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=[keras.metrics.MeanSquaredError(), Pearson(return_dict=True)])
    return model