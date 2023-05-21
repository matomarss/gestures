import os

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import matplotlib
import seaborn as sns

from data_extraction import load_preprocessing_experiment_results, load_best_accuracies_after_hyper_parameter_search, load_kernel_experiment_results_for_pca_table
from train_classifier import SvmModel, RFModel


def create_preprocessing_experiment_graphs(n, model):
    """
    Create graphs for the results of the experiment on preprocessing combinations
    """
    data = load_preprocessing_experiment_results(n, model)
    for prep in ["None", "center_norm"]:
        x = []
        y_stand = []
        y_minmax = []
        plt.clf()
        if prep == "None":
            plt.title("Bez vlastného predspracovania")
        elif prep == "center_norm":
            plt.title("S center-norm predspracovaním")
        for point in data.keys():
            if point == -1:
                pass
            else:
                x.append(int(point))
                records = data.get(point)
                for rec in records:
                    if rec.get("preprocessing") == prep:
                        if rec.get("scaler") == "StandardScaler()":
                            y_stand.append(rec.get("validation_accuracy"))
                        elif rec.get("scaler") == "MinMaxScaler()":
                            y_minmax.append(rec.get("validation_accuracy"))

        plt.scatter(x, y_stand, color="orange", label="S použitím PCA a štandardného škálovania")
        plt.scatter(x, y_minmax, color="green", label="S použitím PCA a min-max škálovania")

        records = data.get(-1)
        for rec in records:
            if rec.get("preprocessing") == prep:
                if rec.get("scaler") == "StandardScaler()":
                    plt.axhline(y=rec.get("validation_accuracy"), color='orange', linestyle='-', label="S použitím štandardného škálovania bez PCA")
                elif rec.get("scaler") == "MinMaxScaler()":
                    plt.axhline(y=rec.get("validation_accuracy"), color='green', linestyle='-', label="S použitím min-max škálovania bez PCA")
                elif rec.get("scaler") == "None":
                    plt.axhline(y=rec.get("validation_accuracy"), color='black', linestyle='-', label="Bez PCA a škálovania")

        plt.xticks(x, x)
        plt.tick_params(axis='x', labelsize=6)
        plt.xlabel("Počet ponechaných komponentov po PCA")
        plt.ylabel("Validačná presnosť")
        plt.legend()
        #matplotlib.use("pgf")
        #plt.show()

        graph_direc = "graphs"
        if not os.path.exists(graph_direc):
            os.makedirs(graph_direc)

        filename = model.get_name() + f"_major_graph_{prep}_n=" + str(n) + ".jpg"
        plt.savefig(os.path.join(graph_direc, filename), dpi=200, quality=100, format='jpg')


def create_classifiers_comparison_graph():
    #matplotlib.use("pgf")
    plt.clf()
    for model in [SvmModel({}), RFModel({})]:
        data = load_best_accuracies_after_hyper_parameter_search(model)

        x = []
        y = []

        plt.title("Graf vývoja najlepšej dosiahnutej validačnej presnosti vzhľadom na dĺžku sekvencie", fontsize = 8)
        for n in data.keys():
            x.append(int(n))
            val_acc = data.get(n)
            y.append(val_acc)

        if model.get_name() == "SVM":
            color = "orange"
            label = "Metóda podporných vektorov (SVM)"
        else:
            color = "green"
            label = "Náhodný les (Random forest)"
        plt.scatter(x, y, color=color)
        plt.plot(x, y, color=color, label=label)

        # for i, point in enumerate(zip(x, y)):
        #     plt.annotate(f'({x[i]})', xy=point, xytext=(-10, 4), textcoords='offset points')

        plt.xticks(x, x)
        plt.xlabel("Dĺžka sekvencie")
        plt.ylabel("Najlepšia dosiahnutá validačná presnosť")

    # Best average validation accuracy achieved by DeepGRU
    plt.axhline(y=0.8074679224679225, color='blue', linestyle='-',
                label="DeepGRU")

    plt.legend()
    #plt.show()
    graph_direc = "graphs"
    if not os.path.exists(graph_direc):
        os.makedirs(graph_direc)

    filename = "Final_graph.jpg"

    plt.savefig(os.path.join(graph_direc, filename), dpi=200, quality=100, format='jpg')


def get_table(data):
    # Extract the columns name and values
    cols = list(data.values())[0].keys()
    rows = data.keys()

    # Prepare the data in a tabular format
    tabular_data = [[data[row][col] for col in cols] for row in rows]
    print(tabulate(tabular_data, headers=cols, tablefmt='latex'))


def visualize_confusion_matrix(cm):
    labels = ['ukázať', 'uchopiť', 'mávať', 'pýtať si', 'ok', 'nič']
    # plot the confusion matrix as a heatmap
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)

    # add labels and title
    plt.xlabel('Skutočné gesto')
    plt.ylabel('Predikované gesto')
    plt.title('Matica zámen')

    graph_direc = "graphs"
    if not os.path.exists(graph_direc):
        os.makedirs(graph_direc)

    filename = "cm.jpg"

    plt.savefig(os.path.join(graph_direc, filename), dpi=200, quality=100, format='jpg')

    # show the plot
    # plt.show()


if __name__ == '__main__':
    print(get_table(load_kernel_experiment_results_for_pca_table("None")))
    print(get_table(load_kernel_experiment_results_for_pca_table(42)))
    print(get_table(load_kernel_experiment_results_for_pca_table(360)))

    for n in [1, 10, 20, 40]:
        for mod in [SvmModel({}), RFModel({})]:
            create_preprocessing_experiment_graphs(n, mod)

    create_classifiers_comparison_graph()



