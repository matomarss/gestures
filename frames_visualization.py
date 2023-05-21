import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_features(file_path):
    """
    Loads feature vectors from the dataset
    """
    data = np.load(file_path)
    features = data['features']
    return features


def visualize_data_frames(file_path):
    """
        Plot the first 40 frames of a gesture recording at a given file path
    """
    features = load_features(file_path)

    for i in range(40):
        frame = features[i]

        palm = frame[:3]
        fingers = frame[3:]

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(fingers[0::3], fingers[1::3], fingers[2::3])
        ax.scatter3D(palm[0], palm[1], palm[2])

        for i in range(5):
            x = np.array([palm[0], fingers[i*3]])
            y = np.array([palm[1], fingers[i*3+1]])
            z = np.array([palm[2], fingers[i*3+2]])
            ax.plot(x, y, z, 'gray')

        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='path to the individual data files in .npz format')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # visualize data frames from the recording at a given path in the dataset
    # please enter the path to the recording in your dataset location in the argument --file_path
    visualize_data_frames(args.file_path)
