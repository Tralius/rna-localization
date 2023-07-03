from typing import Dict, List
from collections import Counter
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, MultiHeadAttention, Reshape, LeakyReLU, \
    BatchNormalization, Concatenate, add
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_probability as tfp


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
    plt.rcParams["figure.figsize"] = (20, 10)
    for dataset in data:
        plt.plot(dataset)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend, loc='upper left')
    plt.show()


def box_plot(dataframe):
    loc_data = dataframe.iloc[:, 0:9]

    sum_vec = loc_data.sum(axis=1)

    loc_data = loc_data.divide(sum_vec, axis='index')

    loc_data_dict = {}
    for loc in loc_data.head():
        loc_data_dict[loc] = loc_data[loc]

    new_loc_data = pd.DataFrame(loc_data_dict)

    plt.rcParams["figure.figsize"] = (20, 10)

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


# https://towardsdatascience.com/how-to-calculate-roc-auc-score-for-regression-models-c0be4fdf76bb


def roc_curve_plot(testY, predictedY):

    testY = testY.iloc[:, 0:9]
    sum_vec = testY.sum(axis=1)
    testY = testY.divide(sum_vec, axis='index')

    classes = list(testY.columns)

    fpr = dict()
    tpr = dict()
    auc_score = []

    y_label_hot_encoding = list()
    pred_label_hot_encoding = list()

    for max_label in testY.idxmax(axis=1):
        one_hot = np.zeros(9)
        one_hot[classes.index(max_label)] = 1
        y_label_hot_encoding.append(one_hot)

    for prediction in predictedY:
        one_hot = np.zeros(9)
        max_label = np.argmax(prediction)
        one_hot[max_label] = 1
        pred_label_hot_encoding.append(one_hot)

    testY = np.array(y_label_hot_encoding)
    predictedY = np.array(pred_label_hot_encoding)

    plt.rcParams["figure.figsize"] = (20,10)

    # calculate the roc_curve for every class
    for i, location in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(testY[:, i], predictedY[:, i])
        auc_score.append(auc(fpr[i], tpr[i]))

    # plot model roc curve
    colors = [
        "blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive"
    ]
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], color=colors[i], marker='.', label=f"{classes[i]}: {round(auc_score[i],2)}")
    # axis labels

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend(loc="lower right", fontsize="25")
    # show the plot
    plt.show()

def scatter_plot(ground_truth, pred):

    ground_truth = ground_truth.iloc[:, 0:9]
    sum_vec = ground_truth.sum(axis=1)
    ground_truth = ground_truth.divide(sum_vec, axis='index')

    classes = list(ground_truth.columns)

    # for i, loc in enumerate(classes):
    '''True label - predicted label scatter'''
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.title(classes[0])
    plt.xlabel('True localization value')
    plt.ylabel('Predicted localization value')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(ground_truth[classes[0]], pred[:, 0], label="")
    plt.legend()


def tf_pearson(y_true, y_pred, sample_axis=0,
                event_axis=None,
                keepdims=False,
                eps=0.001):
    y_true /= (tfp.stats.stddev(y_true, sample_axis=sample_axis, keepdims=True)+eps)
    if y_pred is not None:
      y_pred /= (tfp.stats.stddev(y_pred, sample_axis=sample_axis, keepdims=True)+eps)

    return tfp.stats.covariance(
        x=y_true,
        y=y_pred,
        event_axis=event_axis,
        sample_axis=sample_axis,
        keepdims=keepdims)


def save_plot(path):
    plt.savefig(path)
