import numpy as np
import pandas as pd
from keras.utils import Sequence
from keras.utils import pad_sequences


class GeneDataLoader(Sequence):
    def __init__(self, data_table, batch_size, shuffle=True):
        self.data = data_table
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data))
        self.max_len = self.data['seq'].apply(lambda x: len(x)).max()

        # Calculate the number of batches
        self.num_batches = len(self.data) // batch_size

        if len(self.data) % batch_size != 0:
            self.num_batches += 1

        # Shuffle the indices if enabled
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        # Calculate the start and end indices for the current batch
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        # Adjust the end index if it exceeds the total number of samples
        if end_index > len(self.data):
            end_index = len(self.data)

        # Initialize empty arrays for samples and labels
        max_length = self.data['seq'].apply(lambda x: len(x)).max()
        padded_sequences = np.zeros((end_index - start_index, max_length, 4), dtype=np.float32)
        batch_labels = np.zeros((end_index - start_index, 9), dtype=np.float32)

        # Load padded sequences and labels for the current batch
        for i, idx in enumerate(self.indices[start_index:end_index]):
            sequence = self.data['seq'].iloc[idx]
            label = self.data.iloc[idx, 0:9].values

            #gene_sequence = [eval(vector) for vector in sequence.split(',')]
            #gene_sequence = [eval(vector) for vector in sequence]
            #padded_sequence = pad_sequences([gene_sequence], maxlen=max_length, padding='post', dtype=np.float32)[0]
            padded_sequences[i, :sequence.shape[0], :] = sequence

            #padded_sequences[i] = padded_sequence
            batch_labels[i] = label

        return padded_sequences, batch_labels
