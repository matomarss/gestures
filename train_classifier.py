import argparse
import os
import pickle

import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.svm import SVC

from numba import jit, cuda

import math
import json

from sklearn.model_selection import GridSearchCV, PredefinedSplit


class Model:
    def __init__(self, hyp_params):
        self.hyp_params = hyp_params

    def get_model(self, params):
        pass

    def get_name(self):
        pass

    def get_hyper_parameters(self):
        return self.hyp_params.copy()

    def get_file_name(self, n, p, params):
        pass


class SvmModel(Model):
    def get_name(self):
        return "SVM"

    def get_file_name(self, n, p, params):
        return 'SVM_k_{}_n{}_p{}_C{}_g{}.joblib'.format(params.get("svc_kernel"), n, p, params.get("svc__C"), params.get("svc__gamma"))

    # def get_hyper_parameters(self):
    #     hyp_params = {'svc__C': [math.pow(2, -3),math.pow(2, 3),math.pow(2, 7)], 'svc__kernel': ["rbf"]}
    #     #hyp_params = {'svc__C': [math.pow(2, 3),  math.pow(2, 7)], 'svc__gamma': [0.135, 0.250]}
    #     #hyp_params = {'svc__C': [math.pow(2, -5),  math.pow(2, -1),  math.pow(2, 3),  math.pow(2, 7),  math.pow(2, 11), math.pow(2, 15)], 'svc__gamma': [math.pow(2, 5),  math.pow(2, 1),  math.pow(2, -3),  math.pow(2, -7),  math.pow(2, -11), math.pow(2, -15)]}
    #     return hyp_params

    def get_model(self, params):
        if len(params) == 0:
            svm = SVC(kernel="rbf")
        else:
            if params.get("svc__C") is None:
                C=1
            else:
                C = params.get("svc__C")
            if params.get("svc__gamma") is None:
               gamma="scale"
            else:
                gamma = params.get("svc__gamma")
            if params.get("svc__kernel") is None:
               kernel="rbf"
            else:
                kernel = params.get("svc__kernel")
            svm = SVC(kernel=kernel, C=C, gamma=gamma)
        return svm


class RFModel(Model):
    def get_name(self):
        return "RANDOM FOREST"

    def get_file_name(self, n, p, params):
        return 'RF_n{}_p{}_est{}_d{}.joblib'.format(n, p, params.get("randomforestclassifier__n_estimators"), params.get("randomforestclassifier__max_depth"))

    # def get_hyper_parameters(self):
    #     #hyp_params = dict(randomforestclassifier__n_estimators=[100], randomforestclassifier__max_depth=[150])
    #     #hyp_params = {'randomforestclassifier__n_estimators': [100, 300, 600, 1000], 'randomforestclassifier__max_depth': [50, 100, 225, 500]}
    #     hyp_params = {'randomforestclassifier__max_depth': [50]}
    #     return hyp_params

    def get_model(self, params):
        if len(params) == 0:
            forest = RandomForestClassifier()
        else:
            if params.get("randomforestclassifier__n_estimators") is None:
                n_estimators = 100
            else:
                n_estimators = params.get("randomforestclassifier__n_estimators")
            forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=params.get("randomforestclassifier__max_depth"))
        return forest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_frames', nargs="*", type=int, default=[1], help='How many frames to use (list of all values to try)')
    parser.add_argument('-p', '--preprocessing', type=str, default=None, help='What preprocessing to use')
    parser.add_argument('-m', '--model', type=str, default="svm", help='What machine learning model to use')
    parser.add_argument('root_path', help='path to the individual data files in .npz format')
    args = parser.parse_args()
    return args


def create_seq_data(features, targets, n=5):
    np.set_printoptions(threshold=np.inf)

    features = np.column_stack([features[i: len(features) - n + i, :] for i in range(n)])
    targets = targets[n:]

    return features, targets


debugbol = False


def make_preprocessing(features, targets, preprocessing, n):
    if preprocessing == "center":
        new_features = []
        for i in range(len(features)):
            palm_co_x = features[i][0]
            palm_co_y = features[i][1]
            palm_co_z = features[i][2]
            new_features.append([features[i][0], features[i][1], features[i][2]])
            for j in range(3, len(features[i]), 3):
                new_features[i].append(features[i][j] - palm_co_x)
                new_features[i].append(features[i][j+1] - palm_co_y)
                new_features[i].append(features[i][j+2] - palm_co_z)
        new_features = np.array(new_features)

        return new_features, targets
    if preprocessing == "track":
        new_features = [0]*24
        previous = []
        for i in range(len(features)):
            if i == 0:
                previous = features[i]
                continue
            for e in range(6):
                new_features[e*4] += features[i][e*3] - previous[e*3]
                new_features[e*4+1] += features[i][e*3+1] - previous[e*3+1]
                new_features[e*4+2] += features[i][e*3+2] - previous[e*3+2]
                new_features[e*4+3] += math.dist(features[i][e*3:e*3+3], previous[e*3:e*3+3])
            previous = features[i]
        new_targets = targets[:1]
        return new_features, new_targets
    if preprocessing == "distance":
        global debugbol

        new_features = []
        for i in range(len(features)):
            frame_vector = []
            for j in range(6):
                feature_vector = features[i][j*3:j*3+3]
                for k in range((j+1)*3, len(features[i]), 3):
                    other_feature_vector = features[i][k:k+3]
                    frame_vector.append(math.dist(feature_vector, other_feature_vector))
                    if not debugbol:
                        print(feature_vector)
                        print(other_feature_vector)
                        print(math.dist(feature_vector, other_feature_vector))
            if not debugbol:
                print(features[i])
                print(frame_vector)
                debugbol = True

            new_features.append(frame_vector)
        new_features = np.array(new_features)

        return new_features, targets
    if preprocessing == "distance2":
        new_features = []
        for i in range(len(features)):
            frame_vector = np.copy(features[i])
            for j in range(6):
                feature_vector = features[i][j * 3:j * 3 + 3]
                for k in range((j + 1) * 3, len(features[i]), 3):
                    other_feature_vector = features[i][k:k + 3]
                    frame_vector = np.append(frame_vector, [math.dist(feature_vector, other_feature_vector)])
            new_features.append(frame_vector)
        new_features = np.array(new_features)
        return new_features, targets
    if preprocessing == "center_norm":
        new_features = []
        #print(features[0])
        for i in range(len(features)):

            palm_co_x = features[i][0]
            palm_co_y = features[i][1]
            palm_co_z = features[i][2]
            middle_co_x = features[i][9] - palm_co_x
            middle_co_y = features[i][10] - palm_co_y
            middle_co_z = features[i][11] - palm_co_z
            normalizer = math.sqrt(middle_co_x ** 2 + middle_co_y ** 2 + middle_co_z ** 2)
            new_features.append([features[i][0]/200, features[i][1]/200, features[i][2]/200])
            for j in range(3, len(features[i]), 3):
                new_features[i].append((features[i][j] - palm_co_x)/normalizer)
                new_features[i].append((features[i][j + 1] - palm_co_y)/normalizer)
                new_features[i].append((features[i][j + 2] - palm_co_z)/normalizer)
        new_features = np.array(new_features)
        #print(new_features[0])

        return new_features, targets
    raise Exception("Incorrect PREPROCESSING")


def allows_sequencing(preprocessing):
    if preprocessing in {None, "center", "distance", "distance2", "center_norm"}:
        return True
    return False


def load_data(root_path, cond, preprocessing=None, n=5):
    participant_dir = [f for f in os.listdir(root_path) if cond(f)]

    feature_list = []
    target_list = []

    for participant in participant_dir:
        npz_filenames = os.listdir(os.path.join(root_path, participant))
        for npz_filename in npz_filenames:
            data = np.load(os.path.join(root_path, participant, npz_filename))

            features = data['features']
            targets = data['targets']

            if preprocessing is not None:
                features, targets = make_preprocessing(features, targets, preprocessing, n=n)
            if allows_sequencing(preprocessing):
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


def test_cond(x):
    return '2_zuzka' in x or '2_stefan' in x or '2_palo' in x


def train_cond(x):
    return not test_cond(x)


def true_cond(x):
    return True


def pca_n_components_to_try(data):
    return [5, 15, 40, len(data[0])-1]


def train_and_evaluate(mod, root_path, preprocessing, n, scaler, use_pca, cv=None, pca_n_components_to_try=None):
    print(f"---Training {mod.get_name()} classifier---")
    print(f"Preprocessing = {preprocessing}")
    print(f"Scaler = {scaler}")

    if use_pca:
        pca = PCA()
        print(f"PCA used")
    else:
        pca = None
        print(f"PCA NOT used")

    best_params = {}

    print(f"--Trying n = {n}--")

    train_X, train_y = load_data(root_path, train_cond, n=n, preprocessing=preprocessing)
    test_X, test_y = load_data(root_path, test_cond, n=n, preprocessing=preprocessing)

    print(f"Loaded data with {len(train_X)} training samples and {len(test_X)} testing samples")

    got_model = mod.get_model({})

    pipeline = make_pipeline(scaler, pca, got_model)

    hyp_params = mod.get_hyper_parameters()
    if use_pca:
        if pca_n_components_to_try is None:
            hyp_params["pca__n_components"] = [len(train_X[0]) - 1]
        else:
            hyp_params["pca__n_components"] = pca_n_components_to_try

    if cv is None:
        split_n = 14 / 15
        split = PredefinedSplit(
            [0] * math.ceil(len(train_X) * (1 - split_n)) + [-1] * math.floor(len(train_X) * split_n))
    else:
        split = cv
    search = GridSearchCV(pipeline, hyp_params, verbose=3, cv=split, n_jobs=-1)

    search.fit(train_X, train_y)

    print("Accuracy after the validation of the best model: {}".format(search.best_score_))
    print(f"Hyperparameters of the best model:")
    for name in hyp_params.keys():
        best_val = search.best_params_.get(name)
        best_params[name] = best_val

        print(f"->{name} = {best_val}")

    print("Fitting of the best model in progress...")
    got_model = mod.get_model(best_params)

    pipeline = make_pipeline(scaler, pca, got_model)
    pipeline.fit(train_X, train_y)

    test_predict = pipeline.predict(test_X)

    acc = accuracy_score(test_y, test_predict)
    cm = confusion_matrix(test_y, test_predict)

    print("Accuracy of the best model: {}".format(acc))
    print("Confusion matrix of the best model:")
    print(cm)

    return pipeline, search.cv_results_, [search.best_params_, search.best_score_]


        # if not os.path.exists('models'):
        #     os.makedirs('models')
        #
        # joblib.dump(pipeline, os.path.join('models', mod.get_file_name(i, preprocessing, best_params)))
        #
        # if not os.path.exists('results'):
        #     os.makedirs('results')
        #
        # filename = dump_to_json(mod, preprocessing, scaler, pca, acc, i, search.best_params_.get("svc__kernel"), search.best_params_.get("pca__n_components"))
        # joblib.dump(search.cv_results_, os.path.join('results', filename) + ".joblib")
        # joblib.dump(pipeline,           os.path.join('models', filename) + ".joblib")


# def dump_to_json(model, preprocessing, scaler, pca, accuracy, length, kernel, n_components):
#     # Create a dictionary to store the information
#     data = {
#         "kernel": str(kernel),
#         "model": str(model),
#         "preprocessing": str(preprocessing),
#         "scaler": str(scaler),
#         "pca": str(pca),
#         "length": length,
#         "accuracy": accuracy
#     }
#
#     # Create a filename based on the parameter names and values
#     filename = f"kernel_{kernel}_model_{model}_preprocessing_{preprocessing}_scaler_{scaler}_pca_{pca}_len_{length}_n_components_{n_components}_acc_{accuracy}"
#     path = os.path.join('results', filename) + ".json"
#     # Save the dictionary to a JSON file
#     with open(path, "w") as f:
#         json.dump(data, f)
#
#     return filename


# def test_model(mod, root_path, preprocessing, n, scaler, use_pca, hyp_params):
#     if mod == "svm":
#         m = SvmModel()
#     elif mod == "rf":
#         m = RFModel()
#     else:
#         raise Exception("Incorrect MODEL selected")
#
#     print(f"---Training {m.get_name()} classifier---")
#     print(f"Preprocessing = {preprocessing}")
#     print(f"Scaler = {scaler}")
#
#     if use_pca:
#         pca = PCA()
#         print(f"PCA used")
#     else:
#         pca = None
#         print(f"PCA NOT used")
#
#     print("Hyperparameters:")
#     for name in hyp_params.keys():
#         print(f"->{name} = {hyp_params.get(name)}")
#
#     print(f"--n = {n}--")
#
#     train_X, train_y = load_data(root_path, train_cond, n=n, preprocessing=preprocessing)
#     test_X, test_y = load_data(root_path, test_cond, n=n, preprocessing=preprocessing)
#
#     print(f"Loaded data with {len(train_X)} training samples and {len(test_X)} testing samples")
#
#     print("Fitting of the model in progress...")
#     got_model = m.get_model(hyp_params)
#
#     pipeline = make_pipeline(scaler, pca, got_model)
#     pipeline.fit(train_X, train_y)
#
#     test_predict = pipeline.predict(test_X)
#
#     acc = accuracy_score(test_y, test_predict)
#     cm = confusion_matrix(test_y, test_predict)
#
#     print("Accuracy of the model: {}".format(acc))
#     print("Confusion matrix of the model:")
#     print(cm)


# def print_difference(val_y, val_predict):
#     print("val_predict ----- val_y")
#     for i in range(len(val_y)):
#         print(val_predict[i], "-----", val_y[i])
#
#
# def train_and_evaluate_linear_classification(root_path, n=1):
#     # cond_val = lambda x: '2' in x
#     # cond_train = lambda x: '0' in x or '1' in x
#     cond_val = lambda x: 'jano' in x or 'zuzka' in x or 'iveta' in x # or 'stefan' in x or 'palo' in x
#     # cond_val = lambda x: 'viktor' in x #or 'zuzka' in x or 'iveta' in x or 'stefan' in x or 'palo' in x
#     cond_train = lambda x: not cond_val(x)
#
#     train_X, train_y = load_data(root_path, cond_train, n=n)
#     val_X, val_y = load_data(root_path, cond_val, n=n)
#
#     print(f"Loaded data with {len(train_X)} training samples and {len(val_X)} test samples")
#
#     sgd = SGDClassifier()
#     sgd.fit(train_X, train_y)
#
#     val_predict = sgd.predict(val_X)
#
#     print_difference(val_y, val_predict)
#
#     acc = accuracy_score(val_y, val_predict)
#
#     print("Accuracy: {}".format(acc))
#     print(classification_report(val_y, val_predict))


if __name__ == '__main__':
    args = parse_args()

    # train_and_evaluate("rf", args.root_path, n=[20], preprocessing=None, scaler=MinMaxScaler(), use_pca=True, cv=5)
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing=None, scaler=MinMaxScaler(), use_pca=True, cv=5)
    #
    # train_and_evaluate("rf", args.root_path, n=[20], preprocessing=None, scaler=StandardScaler(), use_pca=True, cv=5)
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing=None, scaler=StandardScaler(), use_pca=True, cv=5)
    #
    # train_and_evaluate("rf", args.root_path, n=[20], preprocessing="center_norm", scaler=MinMaxScaler(), use_pca=True, cv=5)
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing="center_norm", scaler=MinMaxScaler(), use_pca=True, cv=5)
    #
    # train_and_evaluate("rf", args.root_path, n=[20], preprocessing="center_norm", scaler=StandardScaler(), use_pca=True, cv=5)
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing="center_norm", scaler=StandardScaler(), use_pca=True, cv=5)

    # train_and_evaluate("rf", args.root_path, n=[20], preprocessing=None, scaler=MinMaxScaler(), use_pca=False, cv=5)
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing=None, scaler=MinMaxScaler(), use_pca=False, cv=5)
    #
    # train_and_evaluate("rf", args.root_path, n=[20], preprocessing=None, scaler=StandardScaler(), use_pca=False, cv=5)
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing=None, scaler=StandardScaler(), use_pca=False, cv=5)
    #
    # train_and_evaluate("rf", args.root_path, n=[20], preprocessing="center_norm", scaler=MinMaxScaler(), use_pca=False,
    #                    cv=5)
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing="center_norm", scaler=MinMaxScaler(), use_pca=False,
    #                    cv=5)
    #
    # train_and_evaluate("rf", args.root_path, n=[20], preprocessing="center_norm", scaler=StandardScaler(), use_pca=False,
    #                    cv=5)
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing="center_norm", scaler=StandardScaler(),
    #                    use_pca=False, cv=5)

    #train_and_evaluate("svm", args.root_path, n=[20], preprocessing=None, scaler=MinMaxScaler(), use_pca=False, cv=5)
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing=None, scaler=MinMaxScaler(), use_pca=True, cv=5)
    #
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing=None, scaler=StandardScaler(), use_pca=False, cv=5)
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing=None, scaler=StandardScaler(), use_pca=True, cv=5)
    #
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing="center_norm", scaler=MinMaxScaler(), use_pca=False, cv=5)
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing="center_norm", scaler=MinMaxScaler(), use_pca=True, cv=5)
    #
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing="center_norm", scaler=StandardScaler(), use_pca=False,
    #                    cv=5)
    # train_and_evaluate("svm", args.root_path, n=[20], preprocessing="center_norm", scaler=StandardScaler(), use_pca=True,
    #                    cv=5)

    # test_model("svm", args.root_path, n=40, preprocessing=None, scaler=MinMaxScaler(), use_pca=True, hyp_params={'svc__C': 2048, 'svc__gamma': 0.00048828125})
    # test_model("svm", args.root_path, n=40, preprocessing=None, scaler=MinMaxScaler(), use_pca=True, hyp_params={'svc__C': math.pow(2, 3), 'svc__gamma': 0.125})

    # train_and_evaluate("rf", args.root_path, n=[10, 30, 50], preprocessing="center_norm", scaler=None, use_pca=False)
    # train_and_evaluate("svm", args.root_path, n=[40], preprocessing="center_norm", scaler=None, use_pca=False)
    # train_and_evaluate("rf", args.root_path, n=[10, 30, 50], preprocessing="center_norm", scaler=MinMaxScaler(), use_pca=False)
    #train_and_evaluate("svm", args.root_path, n=[40], preprocessing="center_norm", scaler=MinMaxScaler(), use_pca=False, cv=10)
    # train_and_evaluate("rf", args.root_path, n=[10, 30, 50], preprocessing=None, scaler=MinMaxScaler(), use_pca=False)
    # train_and_evaluate("svm", args.root_path, n=[40], preprocessing=None, scaler=MinMaxScaler(), use_pca=False)
    # train_and_evaluate("rf", args.root_path, n=[10, 30, 50], preprocessing=None, scaler=MinMaxScaler(), use_pca=True)
    # train_and_evaluate("svm", args.root_path, n=[40], preprocessing=None, scaler=MinMaxScaler(), use_pca=True)
    #
    # train_and_evaluate("rf", args.root_path, n=[10, 30, 50], preprocessing="center_norm", scaler=StandardScaler(), use_pca=False)
    # train_and_evaluate("svm", args.root_path, n=[40], preprocessing="center_norm", scaler=StandardScaler(), use_pca=False)
    # train_and_evaluate("rf", args.root_path, n=[10, 30, 50], preprocessing=None, scaler=StandardScaler(), use_pca=False)
    # train_and_evaluate("svm", args.root_path, n=[40], preprocessing=None, scaler=StandardScaler(), use_pca=False)
    # train_and_evaluate("rf", args.root_path, n=[10, 30, 50], preprocessing=None, scaler=StandardScaler(), use_pca=True)
    # train_and_evaluate("svm", args.root_path, n=[40], preprocessing=None, scaler=StandardScaler(), use_pca=True)
    #
    # train_and_evaluate("rf", args.root_path, n=[10, 30, 50], preprocessing="center_norm", scaler=None, use_pca=True)
    # train_and_evaluate("svm", args.root_path, n=[40], preprocessing="center_norm", scaler=None, use_pca=True)


    #train_and_evaluate(args.model, args.root_path, n=args.num_frames, preprocessing=args.preprocessing)
