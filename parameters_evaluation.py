import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import math

from train_classifier import train_and_evaluate, parse_args, SvmModel, RFModel, load_data, true_cond, test_model
from files_organization import dump_to_json, dump_object
from data_extraction import find_highest_accuracy_at_preprocessing_experiment, load_preprocessing_experiment_results, load_kernel_experiment_results_for_pca_table, load_best_accuracies_after_hyper_parameter_search
from results_visualisation import create_preprocessing_experiment_graphs, create_classifiers_comparison_graph, get_table, visualize_confusion_matrix


def run_svm_kernel_experiment(root_path):
    """
       Search for the best SVM kernel to be used on the dataset at root_path.
       This experiment is performed for the following combinations of preprocessing for other hyperparameters of SVM
       fixed and for the sequence length fixed to 20:

       NO PCA / number of components left after PCA: 42, 360

       center-norm / without center-norm

       standard scaling / min-max scaling


    """
    if not os.path.exists('test_svm_kernels'):
        os.makedirs('test_svm_kernels')

    # Search the preprocessing options using 5-fold cross-validation
    results = {"linear": [], "poly": [], "rbf": []}
    for preprocessing in [None, "center_norm"]:
        for scaler in [MinMaxScaler(), StandardScaler()]:
            res, best = train_and_evaluate(
                SvmModel({'svc__C': [8], 'svc__kernel': ["rbf", "linear", "poly"]}), root_path, n=20,
                preprocessing=preprocessing, scaler=scaler, use_pca=True, cv=5, pca_n_components_to_try=[42, 20*18])

            for i in range(6):
                record = {
                    "preprocessing": str(preprocessing),
                    "scaler": str(scaler),
                    "pca": res["param_pca__n_components"][i],
                    "validation_accuracy": res["mean_test_score"][i]
                }
                results[res["param_svc__kernel"][i]].append(record)

    for preprocessing in [None, "center_norm"]:
        for scaler in [MinMaxScaler(), StandardScaler()]:
            res, best = train_and_evaluate(SvmModel({'svc__C': [8], 'svc__kernel': ["rbf", "linear", "poly"]}), root_path, n=20,
                                                  preprocessing=preprocessing, scaler=scaler, use_pca=False, cv=5)
            for i in range(3):
                record = {
                    "preprocessing": str(preprocessing),
                    "scaler": str(scaler),
                    "pca": "None",
                    "validation_accuracy": res["mean_test_score"][i]
                }
                results[res["param_svc__kernel"][i]].append(record)

    # Store the results
    dump_to_json("test_svm_kernels", "SVM_kernel_test_n=20", results)


def find_the_number_of_important_components(root_path, n, threshold):
    """
    Find the minimal number of principal components left after PCA preserving at least "threshold" variance in the data
    prepared for sequence length "n". The dataset used is located at "root_path".
    """
    data, y = load_data(root_path=root_path, cond=true_cond, preprocessing=None, n=n)

    # Perform PCA
    pca = PCA()
    pca.fit(data)

    # Get the explained variance ratio
    explained_variance = pca.explained_variance_ratio_

    # Plot the explained variance ratio
    plt.bar(range(len(explained_variance)), explained_variance)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    # plt.show()

    # Find the number of reasonably important principal components
    cumulative_explained_variance = np.cumsum(explained_variance)

    # Find the number of principal components that exceed the threshold
    num_components = np.where(cumulative_explained_variance > threshold)[0][0] + 1
    return num_components


def run_preprocessing_experiment(root_path, n, pca_n_components_to_try):
    """
        For the models SVM and Random Forest, search all the possible combinations of the following preprocessing
        of the dataset from "root_path" using length of the sequence of frames to represent one gesture "n":

        When using PCA:

        number of components left after PCA: pca_n_components_to_try / all components left

        center-norm / without center-norm

        standard scaling / min-max scaling


        When not using PCA:

        center-norm / without center-norm

        standard scaling / min-max scaling / NO scaling
    """
    dir = 'test_pca_n=' + str(n)
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Fix the hyperparameters of the models
    models = [SvmModel({'svc__C': [math.pow(2, 3)]}), RFModel({'randomforestclassifier__max_depth': [50]})]

    # Also verify the option of leaving all principal components after PCA
    pca_n_components_to_try.append(n * 18)

    # Test with PCA used
    for model in models:
        # Initialize the dictionary for the case of all number of components left after PCA
        dictPCA = {"all": []}
        # Initialize dictionaries for the rest of the numbers of components left after PCA
        for n_comp in pca_n_components_to_try:
            if n_comp != n * 18:
                dictPCA[int(n_comp)] = []
        # Test the values for every combination of preprocessing and scaling
        for preprocessing in [None, "center_norm"]:
            for scaler in [MinMaxScaler(), StandardScaler()]:
                res1, best1 = train_and_evaluate(model, root_path, n=n, preprocessing=preprocessing,
                                                         scaler=scaler, use_pca=True, cv=5,
                                                         pca_n_components_to_try=pca_n_components_to_try)
                for i in range(len(pca_n_components_to_try)):
                    data = {
                        "{}".format("C" if "param_svc__C" in res1.keys() else "max_depth"): "{}".format(
                            res1["param_svc__C"][i] if "param_svc__C" in res1.keys() else
                            res1["param_randomforestclassifier__max_depth"][i]),
                        "preprocessing": str(preprocessing),
                        "scaler": str(scaler),
                        "validation_accuracy": res1["mean_test_score"][i]
                    }
                    if res1["param_pca__n_components"][i] == n * 18:
                        dictPCA["all"].append(data)
                    else:
                        dictPCA[int(res1["param_pca__n_components"][i])].append(data)

        # Store the results for the case of using PCA
        filename = model.get_name() + "test_with_PCA_n=" + str(n)
        dump_to_json(dir, filename, dictPCA)

    # Test with PCA NOT used
    for model in models:
        # Initialize the dictionary for the case of not using PCA
        dictNOPCA = {"NO_PCA": []}
        for preprocessing in [None, "center_norm"]:
            for scaler in [MinMaxScaler(), StandardScaler(), None]:
                res2, best2 = train_and_evaluate(model, root_path, n=n, preprocessing=preprocessing,
                                                         scaler=scaler, use_pca=False, cv=5)
                data = {
                    "{}".format("C" if "param_svc__C" in res2.keys() else "max_depth"): "{}".format(
                        res2["param_svc__C"][0] if "param_svc__C" in res2.keys() else
                        res2["param_randomforestclassifier__max_depth"][0]),
                    "preprocessing": str(preprocessing),
                    "scaler": str(scaler),
                    "validation_accuracy": res2["mean_test_score"][0]
                }
                dictNOPCA["NO_PCA"].append(data)

        # Store the results for the case of not using PCA
        filename = model.get_name() + "test_with_NO_PCA_n=" + str(n)
        dump_to_json(dir, filename, dictNOPCA)


def run_hyperparameter_search(mod, root_path, n, preprocessing, scaler, use_pca, n_components):
    """
    Perform the search of the hyperparameters provided with the given model on the data from the dataset at "root_path".

    :param mod: the model to perform the hyperparameter search on
    :param root_path: path to the dataset used
    :param n: length of the sequence representing one gesture
    :param preprocessing: preprocessing to apply to the data
    :param scaler: scaling to apply to the data
    :param use_pca: whether to use PCA
    :param n_components: number of components left after PCA if used
    """
    # Perform the evaluation of the hyperparameters as 5-fold cross-validation
    cv_res, best_result = train_and_evaluate(mod, root_path, preprocessing=preprocessing, n=n, use_pca=use_pca,
                                             pca_n_components_to_try=[n_components], scaler=scaler, cv=5)

    iterations = len(mod.get_hyper_parameters().get("svc__C" if "param_svc__C" in cv_res.keys() else
                                                    "randomforestclassifier__max_depth")) * \
                 len(mod.get_hyper_parameters().get("svc__gamma" if "param_svc__gamma" in cv_res.keys() else
                                                    "randomforestclassifier__n_estimators"))
    # Best accuracy achieved and the hyperparameters it was achieved for
    best = {"best_score": best_result[1]}
    for name in best_result[0].keys():
        best_val = best_result[0].get(name)
        best[name] = best_val
    # Initiaize the dictionary to store the data in
    data = {"results": [], "best_parameters": best}
    # Store the data in the dictionary
    for i in range(iterations):
        data0 = {
            "{}".format("C" if "param_svc__C" in cv_res.keys() else "max_depth"): "{}".format(
                cv_res["param_svc__C"][i] if "param_svc__C" in cv_res.keys() else
                cv_res["param_randomforestclassifier__max_depth"][i]),
            "{}".format("gamma" if "param_svc__gamma" in cv_res.keys() else "n_estimators"): "{}".format(
                cv_res["param_svc__gamma"][i] if "param_svc__gamma" in cv_res.keys() else
                cv_res["param_randomforestclassifier__n_estimators"][i]),
            "preprocessing": str(preprocessing),
            "scaler": str(scaler),
            "pca": use_pca,
            "n_components": n_components,
            "validation_accuracy": cv_res["mean_test_score"][i]
        }
        data["results"].append(data0)

    # Store the results
    dir = "test_hyper_parameters"
    if not os.path.exists(dir):
        os.makedirs(dir)
    filename = mod.get_name() + "_hyper_par_test_n=" + str(n)
    dump_to_json(dir, filename, data)


if __name__ == '__main__':
    """
       This code was written for visualization purposes only as the actual experiments were not performed in a single run. 
       That is because for every experiment, we needed the results from the previous ones to be able to run it. 
       Nevertheless, this code should be able to be run successfully and with the correct results.
    """

    # Change this line to where the dataset is localized in your device
    gestures_path = "C:\Users\matom\OneDrive\Počítač\skola3\gestures_recognition\gestures\prepped"

    # Run the experiment to search for the optimal SVM kernel
    run_svm_kernel_experiment(gestures_path)

    # For each sequence length, find the number of components that need to be left after PCA to preserve at least the
    # given amount of variance in the data (created for the given sequence length)
    n_comp1 = []
    n_comp10 = []
    n_comp20 = []
    n_comp40 = []
    for threshold in [0.5, 0.8, 0.9, 0.95, 0.98, 0.99]:
        n_comp1.append(find_the_number_of_important_components(gestures_path, 1, threshold))
        n_comp10.append(find_the_number_of_important_components(gestures_path, 10, threshold))
        n_comp20.append(find_the_number_of_important_components(gestures_path, 20, threshold))
        n_comp40.append(find_the_number_of_important_components(gestures_path, 40, threshold))
    print(n_comp1)
    print(n_comp10)
    print(n_comp20)
    print(n_comp40)

    # Run the preprocessing experiments for every sequence length from [1,10,20,40] and corresponding PCA components
    run_preprocessing_experiment(gestures_path, 1, n_comp1)
    run_preprocessing_experiment(gestures_path, 10, n_comp10)
    run_preprocessing_experiment(gestures_path, 20, n_comp20)
    run_preprocessing_experiment(gestures_path, 40, n_comp40)

    # Hyperparameters to choose from
    svm = SvmModel({'svc__C': [math.pow(2, 3), math.pow(2, 7), math.pow(2, 11)],
                    'svc__gamma': [math.pow(2, 3), math.pow(2, -3), math.pow(2, -9), math.pow(2, -15)]})
    rf = RFModel({'randomforestclassifier__n_estimators': [100, 300, 500],
                  'randomforestclassifier__max_depth': [50, 100, 300]})

    # For every sequence length and model, run the optimal hyperparameter search on the data
    # preprocessed with the best combination of preprocessing determined in the previous experiments
    run_hyperparameter_search(svm, gestures_path, 1, "center_norm", MinMaxScaler(), False, None)
    run_hyperparameter_search(rf, gestures_path, 1, None, StandardScaler(), True, 1*18)
    run_hyperparameter_search(svm, gestures_path, 10, "center_norm", MinMaxScaler(), False, None)
    run_hyperparameter_search(rf, gestures_path, 10, None, MinMaxScaler(), True, 10*18)
    run_hyperparameter_search(svm, gestures_path, 20, None, StandardScaler(), True, 20*18)
    run_hyperparameter_search(rf, gestures_path, 20, "center_norm", StandardScaler(), True, 116)
    run_hyperparameter_search(svm, gestures_path, 40, None, MinMaxScaler(),True,40*18)
    run_hyperparameter_search(rf, gestures_path, 40, None, StandardScaler(), True, 223)

    # As the random forest achieved the best accuracy
    # train this model with its optimal hyperparameters and preprocessing on the dataset created for sequence length 20
    # and test it on the testing set preprocessed in the same way. Also visualize the confusion matrix
    # of the classification on the testing set
    visualize_confusion_matrix(test_model(mod=RFModel({'randomforestclassifier__n_estimators': 500, 'randomforestclassifier__max_depth': 300}), root_path=gestures_path, preprocessing="center_norm", n=20, scaler=StandardScaler(), use_pca=True, pca_n_components=116))




