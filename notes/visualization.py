import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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