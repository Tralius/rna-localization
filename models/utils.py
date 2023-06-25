from typing import Dict, List
from collections import Counter
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, MultiHeadAttention, Reshape, LeakyReLU, \
    BatchNormalization, Concatenate, add
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        arch.append(MultiHeadAttention(**params.get('attention')[index.get('attention')])(arg, arg))
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
    
# summarize history for accuracy
def plot_line_graph(data, title, ylabel, xlabel, legend):
    # for i in range(len(data)):
    for dataset in data:
        plt.plot(dataset)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend, loc='upper left')
    plt.show()

def scatter_plot(pred, ground_truth):

    def get_classes(dataframe):

        processed_dataframe = dataframe.iloc[:, 0:9]

        sum_vec = processed_dataframe.sum(axis=1)

        processed_dataframe = processed_dataframe.divide(sum_vec, axis='index')

        dataframe_max_values = processed_dataframe.max(axis=1)

        dataframe_max_values_tags = processed_dataframe.idxmax(axis=1)

        return dataframe_max_values, dataframe_max_values_tags

    ground_truth_max_value, ground_truth_class = get_classes(ground_truth)
    pred_max_value, pred_truth_class = get_classes(pred)

    legend = list(ground_truth.columns[0:9])

    plt.scatter(ground_truth_max_value, ground_truth_class, color="purple")

def box_plot(dataframe):

    loc_data = dataframe.iloc[:, 0:9]

    sum_vec = loc_data.sum(axis=1)

    loc_data = loc_data.divide(sum_vec, axis='index')

    loc_data_dict = {}
    for loc in loc_data.head():
        loc_data_dict[loc] = loc_data[loc]

    new_loc_data = pd.DataFrame(loc_data_dict)

    # Plot
    bp = plt.boxplot(
        # A data frame needs to be converted to an array before it can be plotted this way
        np.array(new_loc_data),
        # You can use the column headings from the data frame as labels
        labels=list(loc_data.columns),
        showfliers=False
    )
    # Axis details
    plt.title('Long Jump Finals')
    plt.ylabel('Probability')
    plt.xlabel('Cellular Compartments')

    plt.show()

    