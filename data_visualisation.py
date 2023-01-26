from mpl_toolkits import mplot3d

import argparse
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    args = parse_args()
    visualize(args.file_path)
