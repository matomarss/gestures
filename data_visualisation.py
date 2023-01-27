from mpl_toolkits import mplot3d

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='path to the individual data files in .npz format')
    args = parser.parse_args()
    return args


def load_features(file_path):
    data = np.load(file_path)
    features = data['features']
    return features


def visualize(file_path):
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


def create_pca_test_graphs(data):
    for prep in ["None", "center_norm"]:
        x = []
        y_stand = []
        y_minmax = []
        plt.clf()
        if prep == "None":
            plt.title("Without additional preprocessing")
        elif prep == "center_norm":
            plt.title("With central-normalized preprocessing")
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

        plt.scatter(x, y_stand, color="orange", label="With PCA and standard scaler")
        #plt.plot(x, y_stand, color="green", label="With PCA and standard scaler")
        plt.scatter(x, y_minmax, color="green", label="With PCA and minmax scaler")
        #plt.plot(x, y_minmax, color="orange", label="With PCA and minmax scaler")

        records = data.get(-1)
        for rec in records:
            if rec.get("preprocessing") == prep:
                if rec.get("scaler") == "StandardScaler()":
                    plt.axhline(y=rec.get("validation_accuracy"), color='orange', linestyle='-', label="Without PCA and with standard scaler")
                elif rec.get("scaler") == "MinMaxScaler()":
                    plt.axhline(y=rec.get("validation_accuracy"), color='green', linestyle='-', label="Without PCA and with minmax scaler")
                elif rec.get("scaler") == "None":
                    plt.axhline(y=rec.get("validation_accuracy"), color='black', linestyle='-', label="Without PCA and any scaler")

        plt.xticks(x, x)
        plt.xlabel("Number of components left after PCA")
        plt.ylabel("Best accuracy acquired")
        plt.legend()
        plt.show()


def create_hyper_parameter_test_graph(data):
    x = []
    y = []

    plt.title("Graph of the best acquired accuracy for every sequence length")
    for n in data.keys():
        x.append(int(n))
        val_acc = data.get(n)
        y.append(val_acc)

    plt.scatter(x, y, color="red")
    plt.plot(x, y, color="red")

    plt.xticks(x, x)
    plt.xlabel("Sequence length")
    plt.ylabel("Best accuracy acquired")
    plt.show()


def get_table(data):
    # Extract the columns name and values
    cols = list(data.values())[0].keys()
    rows = data.keys()

    # Prepare the data in a tabular format
    tabular_data = [[data[row][col] for col in cols] for row in rows]
    print(tabulate(tabular_data, headers=cols, tablefmt='latex'))


if __name__ == '__main__':
    args = parse_args()
    visualize(args.file_path)