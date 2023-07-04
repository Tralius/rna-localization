import numpy as np
import pandas as pd
from keras.utils import Sequence, to_categorical
from typing import Tuple


class GeneDataLoader(Sequence):
    def __init__(self, data_table: pd.DataFrame, padding_length: int, batch_size: int = 32, shuffle: bool = True,
                 struct: bool = False, truncate: bool = False, truncate_length: int = 32):

        transformed_data = data_table.dropna()
        transformed_data = output_normalization(transformed_data.iloc[:, 0:9])
        transformed_data = pd.concat([transformed_data, one_hot_emb(data_table['seq'])], axis=1)
        if struct:
            transformed_data = pd.concat([transformed_data, data_table.iloc[:, 10:]], axis=1)
        self.data = transformed_data
        self.struct = struct  # TODO for future use

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(self.data.shape[0])
        self.max_len = padding_length

        self.truncate_len = truncate_length
        self.truncate = truncate

        # Calculate the number of batches
        self.num_batches = self.data.shape[0] // batch_size

        if len(self.data) % batch_size != 0:
            self.num_batches += 1

        # Shuffle the indices if enabled
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Calculate the start and end indices for the current batch
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.data.shape[0])

        # Initialize empty arrays for samples and labels
        if self.truncate:
            sequences = np.zeros((end_index - start_index, self.truncate_len, 4), dtype=np.float32)
            output = np.zeros((end_index - start_index, 9), dtype=np.float32)

            # Load padded sequences and labels for the current batch
            for i, idx in enumerate(self.indices[start_index:end_index]):
                seq_length = len(self.data['seq'].iloc[idx])

                if seq_length < self.truncate_len:
                    sequences[i, -seq_length:, :] = self.data['seq'].iloc[idx]
                else:
                    copy_length = seq_length - self.truncate_len
                    sequences[i] = self.data['seq'].iloc[idx][copy_length:]

                output[i, :] = self.data.iloc[idx, 0:9]

            return sequences, output

        else:
            padded_sequences = np.zeros((end_index - start_index, self.max_len, 4), dtype=np.float32)
            output = np.zeros((end_index - start_index, 9), dtype=np.float32)

            # Load padded sequences and labels for the current batch
            for i, idx in enumerate(self.indices[start_index:end_index]):
                padded_sequences[i, -len(self.data['seq'].iloc[idx]):, :] = self.data['seq'].iloc[idx]

                output[i, :] = self.data.iloc[idx, 0:9]

            return padded_sequences, output


def one_hot_emb(data: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        'A': 0,
        'C': 1,
        'G': 2,
        'T': 3
    }
    one_hot_encode_lam = lambda seq: to_categorical([mapping[x] for x in seq])
    return data.apply(one_hot_encode_lam)


def output_normalization(data: pd.DataFrame) -> pd.DataFrame:
    sum_vec = data.sum(axis=1)
    return data.divide(sum_vec, axis='index')
