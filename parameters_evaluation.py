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

from train_classifier import train_and_evaluate, parse_args, SvmModel, RFModel, load_data, true_cond
from files_organization import dump_to_json, dump_object


def test_svm_kernels(root_path):
    if not os.path.exists('test_svm_kernels'):
        os.makedirs('test_svm_kernels')

    for preprocessing in [None, "center_norm"]:
        for scaler in [MinMaxScaler(), StandardScaler(), None]:
            model1, res1, best1 = train_and_evaluate(SvmModel({'svc__C': [math.pow(2, 3)], 'svc__kernel': ["rbf", "linear", "poly"]}), root_path, n=20, preprocessing=preprocessing,
                                                     scaler=scaler, use_pca=True, cv=5, pca_n_components_to_try=None)
            model2, res2, best2 = train_and_evaluate(SvmModel({'svc__C': [math.pow(2, 3)], 'svc__kernel': ["rbf", "linear", "poly"]}), root_path, n=20, preprocessing=preprocessing,
                                                     scaler=scaler, use_pca=False, cv=5, pca_n_components_to_try=None)
            for i in range(3):
                data1 = {
                    "preprocessing": str(preprocessing),
                    "scaler": str(scaler),
                    "kernel": str(res1["param_svc__kernel"][i]),
                    "validation_accuracy": res1["mean_test_score"][i]
                }
                print(res1["param_svc__kernel"][i])
                print(res1["mean_test_score"][i])
                data2 = {
                    "preprocessing": str(preprocessing),
                    "scaler": str(scaler),
                    "kernel": str(res2["param_svc__kernel"][i]),
                    "validation_accuracy": res2["mean_test_score"][i]
                }
                dump_to_json("test_svm_kernels", "SVMkerneltest_n=20_pcaALL", data1)
                dump_to_json("test_svm_kernels", "SVMkerneltest_n=20_NOpca", data2)
            dump_object("test_svm_kernels", "SVM_kerneltest_n=20_pcaALL", model1)
            dump_object("test_svm_kernels", "SVMkerneltest_n=20_NOpca", model2)


def find_the_number_of_important_components(root_path, n):
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
    plt.show()

    # Find the number of reasonably important principal components
    cumulative_explained_variance = np.cumsum(explained_variance)

    # Set the threshold for the explained variance ratio
    threshold = 0.97

    # Find the number of principal components that exceed the threshold
    num_components = np.where(cumulative_explained_variance > threshold)[0][0] + 1
    print("Number of reasonably important principal components for n = {}:".format(n), num_components)
    print(np.where(cumulative_explained_variance > threshold))
    return num_components


def test_pca(root_path, n, pca_n_components_to_try):
    dir = 'test_pca_n='+str(n)
    if not os.path.exists(dir):
        os.makedirs(dir)
    models = [SvmModel({'svc__C': [math.pow(2, 3)]}), RFModel({'randomforestclassifier__max_depth': [50]})]

    # Also use all possible dimensions
    pca_n_components_to_try.append(n*18)
    # Test with PCA used
    for model in models:
        # Initialize the dictionary for all evaluated number of components
        dictPCA = {"all": []}
        for n_comp in pca_n_components_to_try:
            if n_comp != n*18:
                dictPCA[n_comp] = []
        # Test the values for every combination of preprocessing and scaling
        for preprocessing in [None, "center_norm"]:
            for scaler in [MinMaxScaler(), StandardScaler()]:
                model1, res1, best1 = train_and_evaluate(model, root_path, n=n, preprocessing=preprocessing,
                                                         scaler=scaler, use_pca=True, cv=5, pca_n_components_to_try=pca_n_components_to_try)
                ln = len(pca_n_components_to_try)
                for i in range(ln):
                    data = {
                        "{}".format("C" if "param_svc__C" in res1.keys() else "max_depth"): "{}".format(
                            res1["param_svc__C"][i] if "param_svc__C" in res1.keys() else
                            res1["param_randomforestclassifier__max_depth"][i]),
                        "preprocessing": str(preprocessing),
                        "scaler": str(scaler),
                        "validation_accuracy": res1["mean_test_score"][i]
                    }
                    print(res1["param_pca__n_components"][i])
                    print(data)
                    if res1["param_pca__n_components"][i] == n*18:
                        dictPCA["all"].append(data)
                    else:
                        dictPCA[res1["param_pca__n_components"][i]].append(data)
        filename = model.get_name() + "test_with_PCA_n=" + str(n)
        dump_to_json(dir, filename, dictPCA)
    # Test with PCA NOT used
    for model in models:
        dictNOPCA = {"NO_PCA": []}
        for preprocessing in [None, "center_norm"]:
            for scaler in [MinMaxScaler(), StandardScaler(), None]:
                model2, res2, best2 = train_and_evaluate(model, root_path, n=n, preprocessing=preprocessing,
                                                         scaler=scaler, use_pca=False, cv=5)
                data = {
                    "{}".format("C" if "param_svc__C" in res2.keys() else "max_depth"): "{}".format(
                        res2["param_svc__C"][0] if "param_svc__C" in res2.keys() else
                        res2["param_randomforestclassifier__max_depth"][0]),
                    "preprocessing": str(preprocessing),
                    "scaler": str(scaler),
                    "validation_accuracy": res2["mean_test_score"][0]
                }
                print(data)
                dictNOPCA["NO_PCA"].append(data)
        filename = model.get_name() + "test_with_NO_PCA_n=" + str(n)
        dump_to_json(dir, filename, dictNOPCA)


if __name__ == '__main__':
    abs_path = os.path.abspath("..")
    rel_path = "gestures\prepped"
    path = os.path.join(abs_path, rel_path)
    #test_svm_kernels(os.path.join(abs_path, rel_path))


    # find_the_number_of_important_components(path, 1)
    # find_the_number_of_important_components(path, 10)
    # find_the_number_of_important_components(path, 20)
    # find_the_number_of_important_components(path, 40)

    #99% 1: 12, 10: 62, 20: 116, 40: 223
    #98% 1: 11, 10: 44, 20: 79, 40: 150
    #97% 1: 9, 10: 34, 20: 61, 40: 113
    test_pca(path, 1, [1, 9, 11, 12])
    test_pca(path, 10, [1, 34, 44, 62])
    test_pca(path, 20, [1, 61, 79, 116])
    test_pca(path, 40, [1, 113, 150, 223])
