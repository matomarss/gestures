import os

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import matplotlib
import seaborn as sns

from data_extraction import load_pca_test_results, load_hyper_parameter_test_results
from train_classifier import SvmModel, RFModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='path to the individual data files in .npz format')
    args = parser.parse_args()
    return args


def load_features(file_path):
    data = np.load(file_path)
    features = data['features']
    return features


def load_targets(file_path):
    data = np.load(file_path)
    targets = data['targets']
    return targets


def get_gesture_num(file_path):
    return np.argmax(load_targets(file_path), axis=-1)[0]


def visualize_data(file_path):
    features = load_features(file_path)
    first_frame = features[0]

    palm = first_frame[:3]
    fingers = first_frame[3:]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(fingers[0::3], fingers[1::3], fingers[2::3])
    ax.scatter3D(palm[0], palm[1], palm[2])

    for i in range(5):
        x = np.array([palm[0], fingers[i*3]])
        y = np.array([palm[1], fingers[i*3+1]])
        z = np.array([palm[2], fingers[i*3+2]])
        ax.plot(x, y, z, 'gray')

    show_targets(file_path)

    plt.show()


def show_targets(file_path):
    data = np.load(file_path)
    targets = data['targets']
    print(targets)


def create_pca_test_graph_average(accuracies):
    x = []
    y = []
    accuracies = dict(sorted(accuracies.items()))
    for point in accuracies.keys():
        if point == -1 or point == -2:
            pass
        else:
            x.append(int(point))
            y.append(accuracies.get(point))
    plt.scatter(x, y)
    plt.plot(x, y)

    plt.axhline(y=accuracies.get(-1), color='r', linestyle='-')
    plt.axhline(y=accuracies.get(-2), color='g', linestyle='-')
    plt.show()


def create_pca_best_test_graph(accuracies):
    x = []
    y = []
    accuracies = dict(sorted(accuracies.items()))
    for point in accuracies.keys():
        if point == -1:
            pass
        else:
            x.append(int(point))
            y.append(accuracies.get(point))
    plt.scatter(x, y)
    plt.plot(x, y, label="With PCA")

    plt.xticks(x, x)
    plt.xlabel("Number of components left after PCA")
    plt.ylabel("Best accuracy acquired")

    plt.axhline(y=accuracies.get(-1), color='r', linestyle='-', label="Without PCA")
    plt.legend()
    plt.show()


def create_pca_test_graphs(n, model):
    data = load_pca_test_results(n, model)
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
        #plt.plot(x, y_stand, color="green", label="With PCA and standard scaler")
        plt.scatter(x, y_minmax, color="green", label="S použitím PCA a min-max škálovania")
        #plt.plot(x, y_minmax, color="orange", label="With PCA and minmax scaler")

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


def create_hyper_parameter_test_graph():
    #matplotlib.use("pgf")
    plt.clf()
    for model in [SvmModel({}), RFModel({})]:
        data = load_hyper_parameter_test_results(model)

        x = []
        y = []

        plt.title("Graf vývoja najlepšej dosiahnutej validačnej presnosti vzhľadom na dĺžku sekvencie", fontsize = 8)
        for n in data.keys():
            x.append(int(n))
            val_acc = data.get(n)
            y.append(val_acc)

        if model.get_name() == "SVM":
            color = "red"
            label = "Metóda podporných vektorov (SVM)"
        else:
            color = "blue"
            label = "Náhodný les (Random forest)"
        plt.scatter(x, y, color=color)
        plt.plot(x, y, color=color, label=label)

        # for i, point in enumerate(zip(x, y)):
        #     plt.annotate(f'({x[i]})', xy=point, xytext=(-10, 4), textcoords='offset points')

        plt.xticks(x, x)
        plt.xlabel("Dĺžka sekvencie")
        plt.ylabel("Najlepšia dosiahnutá validačná presnosť")
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
    args = parse_args()
    print(get_gesture_num(args.file_path))
