from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr
import seaborn as sns

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
    
def multiplot_pearson(data: Dict[str, List[float]], mean: bool = True, training: bool = False,title: str = 'Correlation after Compartment'):
    headers = [
               'val_ERM', 'val_KDEL', 'val_LMA', 'val_MITO', 'val_NES', 'val_NIK', 'val_NLS', 'val_NUCP', 'val_OMM'
               ]
    if training:
        headers = ['ERM', 'KDEL', 'LMA', 'MITO', 'NES', 'NIK', 'NLS', 'NUCP', 'OMM'].extend(headers)
    data_list = [data[ind] for ind in headers]
    data_panda = np.array(data_list).transpose()
    if mean:
        mean_vals = np.mean(data_panda, axis=1)
        data_panda = np.hstack([data_panda, np.expand_dims(mean_vals, axis=1)])
        headers.append('mean')
    data_panda = np.hstack([data_panda, np.expand_dims(list(range(len(data['loss']))), axis=1)])
    headers.append('epoch')
    dataframe = pd.DataFrame(data_panda, columns=headers)
    plot_data = dataframe.melt(id_vars=['epoch'], var_name='compartment', value_name='correlation')
    return sns.lineplot(plot_data, x='epoch', y='correlation', hue='compartment').set(title=title)

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


def box_plot_array(multidimensionalarray):
    # Plot
    bp = plt.boxplot(
        # A data frame needs to be converted to an array before it can be plotted this way
        np.array(multidimensionalarray),
        # You can use the column headings from the data frame as labels
        labels=['ERM', 'KDEL', 'LMA', 'MITO', 'NES', 'NIK', 'NLS', 'NUCP', 'OMM'],
        showfliers=False
    )
    # Axis details
    plt.title('Cellular Compoartments Data Distribution')
    plt.ylabel('Probability')
    plt.xlabel('Cellular Compartments')
    plt.show()

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
    plt.title('Cellular Compoartments Data Distribution')
    plt.ylabel('Probability')
    plt.xlabel('Cellular Compartments')

    plt.show()


def bar_plot(ground_truth, pred, mode):
    if mode == "pearson":
        pearson_of_comp = dict()
        compartments = list(ground_truth.columns)

        for i, compartment in enumerate(compartments):
            correlation, pvalue = pearsonr(pred[:, i], ground_truth[compartment])
            pearson_of_comp[compartment] = correlation

        compartments.append("Mean")
        pearson_of_comp["Mean"] = sum(pearson_of_comp.values()) / len(pearson_of_comp)

        plt.rcParams["figure.figsize"] = (20, 10)
        plt.bar(compartments, pearson_of_comp.values(), color='blue', width=0.4)

        plt.xlabel("Compartments")
        plt.ylabel("Pearson Correlation")
        plt.title("Pearson Correlation for Compartments")
        plt.show()
