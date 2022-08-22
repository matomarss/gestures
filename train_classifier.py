import argparse
import os
import pickle

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.svm import SVC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_frames', type=int, default=10, help='How many frames to use')
    parser.add_argument('root_path', help='path to the individual data files in .npz format')
    args = parser.parse_args()
    return args


def create_seq_data(features, targets, n=5):
    features = np.column_stack([features[i: len(features) - n + i, :] for i in range(n)])
    targets = targets[n:]

    return features, targets


def load_data(root_path, cond, n=5):
    participant_dir = [f for f in os.listdir(root_path) if cond(f)]

    feature_list = []
    target_list = []

    for participant in participant_dir:
        npz_filenames = os.listdir(os.path.join(root_path, participant))
        for npz_filename in npz_filenames:
            data = np.load(os.path.join(root_path, participant, npz_filename))

            features = data['features']
            targets = data['targets']

            features, targets = create_seq_data(features, targets, n=n)

            feature_list.append(features)
            target_list.append(targets)


    features = np.row_stack(feature_list)
    targets = np.row_stack(target_list)

    targets = np.argmax(targets, axis=-1)

    return features, targets


def get_multi_pred(y_true, y_pred_single, n=5):
    y_pred_single_one_hot = np.zeros([len(y_pred_single), 6])
    for i, pred in enumerate(y_pred_single):
        y_pred_single_one_hot[i, pred] = 1
    y_pred_multi = np.array([y_pred_single_one_hot[i: len(y_pred_single) - n + i, :] for i in range(n)])
    y_pred_multi = np.sum(y_pred_multi, axis=0)
    y_pred = np.argmax(y_pred_multi, axis=-1)

    y_true = y_true[n:]

    print(f1_score(y_true, y_pred, average='weighted'))




def train_and_evaluate(root_path, n=1):
    # cond_val = lambda x: '2' in x
    # cond_train = lambda x: '0' in x or '1' in x
    cond_val = lambda x: 'jano' in x or 'zuzka' in x or 'iveta' in x # or 'stefan' in x or 'palo' in x
    # cond_val = lambda x: 'viktor' in x #or 'zuzka' in x or 'iveta' in x or 'stefan' in x or 'palo' in x
    cond_train = lambda x: not cond_val(x)

    train_X, train_y = load_data(root_path, cond_train, n=n)
    val_X, val_y = load_data(root_path, cond_val, n=n)

    print(f"Loaded data with {len(train_X)} training samples and {len(val_X)} test samples")

    best_acc = 0.0
    best_d = 1

    for d in range(1, 30):
        pca = PCA()
        forest = RandomForestClassifier(max_depth=d)
        pipeline = make_pipeline(MinMaxScaler(), pca, forest)
        pipeline.fit(train_X, train_y)

        val_predict = pipeline.predict(val_X)

        acc = accuracy_score(val_y, val_predict)

        if acc > best_acc:
            best_acc = acc
            best_d = d

        print("For depth: {}".format(d))
        print("Accuracy: {}".format(acc))
        print(classification_report(val_y, val_predict))

    print("Best ACC for depth: {}".format(best_d))

    true_cond = lambda x : True
    all_X, all_y = load_data(root_path, true_cond, n=n)

    pca = PCA()
    forest = RandomForestClassifier(max_depth=d)
    pipeline = make_pipeline(MinMaxScaler(), pca, forest)
    pipeline.fit(all_X, all_y)

    if not os.path.exists('models'):
        os.makedirs('models')

    filename = 'RF_n{}.joblib'.format(n, d)

    joblib.dump(pipeline, os.path.join('models', filename))



if __name__ == '__main__':
    args = parse_args()
    train_and_evaluate(args.root_path, n=args.num_frames)