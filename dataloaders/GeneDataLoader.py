import numpy as np
import pandas as pd
from keras.utils import Sequence, to_categorical
from typing import Tuple


class GeneDataLoader(Sequence):
    def __init__(self, data_table: pd.DataFrame, padding_length: int, batch_size: int = 32, shuffle: bool = True,
                 struct: bool = False):

        transformed_data = data_table.dropna()
        transformed_data = output_normalization(transformed_data.iloc[:, 0:9])
        transformed_data = pd.concat([transformed_data, one_hot_emb(data_table['seq'])], axis=1)
        if struct:
            transformed_data = pd.concat([transformed_data, data_table['struct']], axis=1)
        self.data = transformed_data
        self.struct = struct

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(self.data.shape[0])
        self.max_len = padding_length

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
        if self.struct:
            padded_sequences = np.zeros((end_index - start_index, self.max_len, 6), dtype=np.float32)
        else:
            padded_sequences = np.zeros((end_index - start_index, self.max_len, 4), dtype=np.float32)
        output = np.zeros((end_index - start_index, 9), dtype=np.float32)

        # Load padded sequences and labels for the current batch
        for i, idx in enumerate(self.indices[start_index:end_index]):
            seq_data = self.data['seq'].iloc[idx]
            output[i, :] = self.data.iloc[idx, 0:9]
            if self.struct:
                tmp = self.data['struct'].iloc[idx]
                padded_mask_struct = np.expand_dims(np.array(tmp != 'nan'), axis=1) #TODO eventuell herausnehmen
                tmp[tmp == 'nan'] = -1
                padded_struct = np.expand_dims(tmp, axis=1).astype('float64')
                seq_data = np.concatenate([seq_data, padded_mask_struct, padded_struct], axis=1)

            padded_sequences[i, -len(self.data['seq'].iloc[idx]):, :] = seq_data
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
