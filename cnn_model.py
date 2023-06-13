import keras
from keras.layers import Conv1D, Dense, Flatten, Input, MaxPooling1D, SpatialDropout1D


class CNN_MLRG:

    def CNN_MLRG(self):
        pass

    def build_model(self):
        model = keras.Sequential()

        model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(4, 64)))
        model.add(MaxPooling1D(pool_size=2))

        # model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
        # model.add(MaxPooling1D(pool_size=2))

        # model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
        # model.add(MaxPooling1D(pool_size=2))

        model.add(Dense(9, activation="softmax"))

        model.add(Flatten())

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

        model.summary()

        return model
