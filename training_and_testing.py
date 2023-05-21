import argparse
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

import math

from sklearn.model_selection import GridSearchCV, PredefinedSplit


class Model:
    """
    Class that represents a classification model. It contains information about its possible hyperparameter value to be
    used in experiments and other useful information.
    """
    def __init__(self, hyp_params):
        self.hyp_params = hyp_params

    def get_model(self, hyperparams):
        """
        Creates a sklearn model with the given hyperparameters
        """
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

    def get_model(self, hyperparams):
        if len(hyperparams) == 0:
            svm = SVC(kernel="rbf")
        else:
            if hyperparams.get("svc__C") is None:
                C=1
            else:
                C = hyperparams.get("svc__C")
            if hyperparams.get("svc__gamma") is None:
               gamma="scale"
            else:
                gamma = hyperparams.get("svc__gamma")
            if hyperparams.get("svc__kernel") is None:
               kernel="rbf"
            else:
                kernel = hyperparams.get("svc__kernel")
            svm = SVC(kernel=kernel, C=C, gamma=gamma)
        return svm


class RFModel(Model):
    def get_name(self):
        return "RANDOM FOREST"

    def get_file_name(self, n, p, params):
        return 'RF_n{}_p{}_est{}_d{}.joblib'.format(n, p, params.get("randomforestclassifier__n_estimators"), params.get("randomforestclassifier__max_depth"))

    def get_model(self, hyperparams):
        if len(hyperparams) == 0:
            forest = RandomForestClassifier()
        else:
            if hyperparams.get("randomforestclassifier__n_estimators") is None:
                n_estimators = 100
            else:
                n_estimators = hyperparams.get("randomforestclassifier__n_estimators")
            forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=hyperparams.get("randomforestclassifier__max_depth"))
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
    """
    Creates a data for the given sequence length.
    :param features: feature vectors of the dataset
    :param targets: targets of the dataset
    :param n: sequence length
    :return: new data for the given sequence length
    """
    features = np.column_stack([features[i: len(features) - n + i, :] for i in range(n)])
    targets = targets[n:]

    return features, targets


def make_preprocessing(features, targets, preprocessing):
    """
    Perform the given preprocessing on the given data.
    :return: preprocessed data
    """
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
        new_features = []
        for i in range(len(features)):
            frame_vector = []
            for j in range(6):
                feature_vector = features[i][j*3:j*3+3]
                for k in range((j+1)*3, len(features[i]), 3):
                    other_feature_vector = features[i][k:k+3]
                    frame_vector.append(math.dist(feature_vector, other_feature_vector))

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
        return new_features, targets

    raise Exception("Incorrect PREPROCESSING")


def allows_sequencing(preprocessing):
    """
    :return: whether the given preprocessing is applicable to data created for sequences
    """
    if preprocessing in {None, "center", "distance", "distance2", "center_norm"}:
        return True
    return False


def load_data(root_path, cond, preprocessing=None, n=5):
    """
    Load the data at "root_path" that satisfy the given condition. Apply preprocessing and create sequences if requested.

    :return loaded data
    """
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
                features, targets = make_preprocessing(features, targets, preprocessing)
            if allows_sequencing(preprocessing):
                features, targets = create_seq_data(features, targets, n=n)

            feature_list.append(features)
            target_list.append(targets)

    features = np.row_stack(feature_list)
    targets = np.row_stack(target_list)

    targets = np.argmax(targets, axis=-1)

    return features, targets


def test_cond(x):
    """
    :return: is this recording supposed to be in the testing set
    """
    return '2_zuzka' in x or '2_stefan' in x or '2_palo' in x


def train_cond(x):
    """
    :return: is this recording supposed to be in the training/validation set
    """
    return not test_cond(x)


def true_cond(x):
    return True


def train_and_evaluate(mod, root_path, preprocessing, n, scaler, use_pca, cv=5, pca_n_components_to_try=None):
    """
    Train and evaluate the given model on the dataset at "root_path".

    :param mod: model to be trained containing the hyperparameters to be evaluated
    :param root_path: path to the dataset
    :param preprocessing: preprocessing to be applied to the training data
    :param n: length of the sequence of frames corresponding to one gesture
    :param scaler: scaling to be performed on the data
    :param use_pca: whether to use PCA
    :param cv: how many folds to use in cross-validation
    :param pca_n_components_to_try: number of principal components left after PCA if used
    :return: cross-validation results as described in sklearn
    """

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

    # load the data
    train_X, train_y = load_data(root_path, train_cond, n=n, preprocessing=preprocessing)
    test_X, test_y = load_data(root_path, test_cond, n=n, preprocessing=preprocessing)

    print(f"Loaded data with {len(train_X)} training samples and {len(test_X)} testing samples")

    got_model = mod.get_model({})

    # make pipeline that applies scaling and PCA before applying the classification model
    pipeline = make_pipeline(scaler, pca, got_model)

    # created the set of hyperparameters to be searched in gridsearch
    hyp_params = mod.get_hyper_parameters()
    if use_pca:
        if pca_n_components_to_try is None:
            hyp_params["pca__n_components"] = [len(train_X[0])]
        else:
            hyp_params["pca__n_components"] = pca_n_components_to_try

    # perform gridsearch for the optimal hyperparameters with the given pipeline and cv-fold cross-validation
    search = GridSearchCV(pipeline, hyp_params, verbose=3, cv=cv, n_jobs=-1)
    search.fit(train_X, train_y)

    print("Accuracy after the validation of the best model: {}".format(search.best_score_))
    print(f"Hyperparameters of the best model:")
    for name in hyp_params.keys():
        best_val = search.best_params_.get(name)
        best_params[name] = best_val

        print(f"->{name} = {best_val}")

    return search.cv_results_, [search.best_params_, search.best_score_]


def test_model(mod, root_path, preprocessing, n, scaler, use_pca, pca_n_components):
    """
    Train the given model on the training set and then test the given model on the testing set created from the dataset
    at "root_path".

    :param mod: The model to be trained and tested with the given hyperparameters
    :param root_path: path to the dataset
    :param preprocessing: preprocessing to be applied to the data
    :param n: length of the sequence of frames corresponding to one gesture
    :param scaler: scaling to be performed on the data
    :param use_pca: whether to use PCA
    :param pca_n_components: number of principal components left after PCA if used
    :return: confusion matrix of the classification at the testing set
    """
    print(f"---Training {mod.get_name()} classifier---")
    print(f"Preprocessing = {preprocessing}")
    print(f"Scaler = {scaler}")

    if use_pca:
        pca = PCA(n_components=pca_n_components)
        print(f"PCA used")
    else:
        pca = None
        print(f"PCA NOT used")

    hyp_params = mod.get_hyper_parameters()
    print("Hyperparameters:")
    for name in hyp_params.keys():
        print(f"->{name} = {hyp_params.get(name)}")

    print(f"--n = {n}--")

    train_X, train_y = load_data(root_path, train_cond, n=n, preprocessing=preprocessing)
    test_X, test_y = load_data(root_path, test_cond, n=n, preprocessing=preprocessing)

    print(f"Loaded data with {len(train_X)} training samples and {len(test_X)} testing samples")

    print("Fitting of the model in progress...")
    # get the model with the given hyperparameters and create a pipeline that will apply preprocessing to the data
    # before using the model
    got_model = mod.get_model(hyp_params)
    pipeline = make_pipeline(scaler, pca, got_model)

    # train the model on the preprocessed training data
    pipeline.fit(train_X, train_y)

    # test the model on the preprocessed testing data
    test_predict = pipeline.predict(test_X)

    acc = accuracy_score(test_y, test_predict)
    cm = confusion_matrix(test_y, test_predict)

    print("Accuracy of the model: {}".format(acc))
    print("Confusion matrix of the model:")
    return cm


if __name__ == '__main__':
    args = parse_args()
