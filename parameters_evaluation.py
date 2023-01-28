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
from data_extraction import find_best_result, load_pca_test_results, load_kernel_test_results, load_hyper_parameter_test_results, load_preprocessing_and_scaler_table_data
from data_visualisation import create_pca_test_graphs, create_hyper_parameter_test_graph, get_table


def test_svm_kernels(root_path):
    if not os.path.exists('test_svm_kernels'):
        os.makedirs('test_svm_kernels')
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

                print(record)

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

                print(record)
    dump_to_json("test_svm_kernels", "SVM_kernel_test_n=20", results)


def find_the_number_of_important_components(root_path, n, threshold):
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
    #print("Number of reasonably important principal components for n = {}:".format(n), num_components)
    #print(np.where(cumulative_explained_variance > threshold))
    return num_components


def test_pca(root_path, n, pca_n_components_to_try):
    dir = 'test_pca_n=' + str(n)
    if not os.path.exists(dir):
        os.makedirs(dir)
    models = [SvmModel({'svc__C': [math.pow(2, 3)]}), RFModel({'randomforestclassifier__max_depth': [50]})]

    # Also use all possible dimensions
    pca_n_components_to_try.append(n * 18)
    # Test with PCA used
    for model in models:
        # Initialize the dictionary for all evaluated number of components
        dictPCA = {"all": []}
        for n_comp in pca_n_components_to_try:
            if n_comp != n * 18:
                dictPCA[int(n_comp)] = []
        # Test the values for every combination of preprocessing and scaling
        for preprocessing in [None, "center_norm"]:
            for scaler in [MinMaxScaler(), StandardScaler()]:
                res1, best1 = train_and_evaluate(model, root_path, n=n, preprocessing=preprocessing,
                                                         scaler=scaler, use_pca=True, cv=5,
                                                         pca_n_components_to_try=pca_n_components_to_try)
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
                    print(int(res1["param_pca__n_components"][i]))
                    print(data)
                    if res1["param_pca__n_components"][i] == n * 18:
                        dictPCA["all"].append(data)
                    else:
                        dictPCA[int(res1["param_pca__n_components"][i])].append(data)
        filename = model.get_name() + "test_with_PCA_n=" + str(n)
        dump_to_json(dir, filename, dictPCA)
    # Test with PCA NOT used
    for model in models:
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
                print(data)
                dictNOPCA["NO_PCA"].append(data)
        filename = model.get_name() + "test_with_NO_PCA_n=" + str(n)
        dump_to_json(dir, filename, dictNOPCA)


def test_hyper_parameters(mod, root_path, n, preprocessing, scaler, use_pca, n_components):
    cv_res, best_result = train_and_evaluate(mod, root_path, preprocessing=preprocessing, n=n, use_pca=use_pca,
                                                   pca_n_components_to_try=[n_components], scaler=scaler, cv=5)
    iterations = len(mod.get_hyper_parameters().get("svc__C" if "param_svc__C" in cv_res.keys() else
                                                    "randomforestclassifier__max_depth")) * \
                 len(mod.get_hyper_parameters().get("svc__gamma" if "param_svc__gamma" in cv_res.keys() else
                                                    "randomforestclassifier__n_estimators"))
    best = {"best_score": best_result[1]}
    for name in best_result[0].keys():
        best_val = best_result[0].get(name)
        best[name] = best_val

    data = {"results": [], "best_parameters": best}
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
    dir = "test_hyper_parameters"
    if not os.path.exists(dir):
        os.makedirs(dir)
    filename = mod.get_name() + "_hyper_par_test_n=" + str(n)
    dump_to_json(dir, filename, data)
    #dump_object(dir, filename, model)


if __name__ == '__main__':
    abs_path = os.path.abspath("..")
    rel_path = "gestures\prepped"
    path = os.path.join(abs_path, rel_path)

    #test_svm_kernels(path)

    # n_comp1 = []
    # n_comp10 = []
    # n_comp20 = []
    # n_comp40 = []
    # for threshold in [0.5, 0.8, 0.9, 0.95, 0.98, 0.99]:
    #     n_comp1.append(find_the_number_of_important_components(path, 1, threshold))
    #     n_comp10.append(find_the_number_of_important_components(path, 10, threshold))
    #     n_comp20.append(find_the_number_of_important_components(path, 20, threshold))
    #     n_comp40.append(find_the_number_of_important_components(path, 40, threshold))
    # print(n_comp1)
    # print(n_comp10)
    # print(n_comp20)
    # print(n_comp40)
    # # 99% 1: 12, 10: 62, 20: 116, 40: 223
    # # 98% 1: 11, 10: 44, 20: 79, 40: 150
    # # 97% 1: 9, 10: 34, 20: 61, 40: 113
    # test_pca(path, 1, n_comp1)
    # test_pca(path, 10, n_comp10)
    # test_pca(path, 20, n_comp20)
    # test_pca(path, 40, n_comp40)

    #print(find_best_result(SvmModel({})))
    #print(find_best_result(RFModel({})))
    # res_rf1 = load_pca_test_results(1, RFModel({}))
    # res_rf10 = load_pca_test_results(10, RFModel({}))
    # res_rf20 = load_pca_test_results(20, RFModel({}))
    # res_rf40 = load_pca_test_results(40, RFModel({}))
    # res_svm1 = load_pca_test_results(1, SvmModel({}))
    # res_svm10 = load_pca_test_results(10, SvmModel({}))
    # res_svm20 = load_pca_test_results(20, SvmModel({}))
    # res_svm40 = load_pca_test_results(40, SvmModel({}))
    # create_pca_test_graphs(res_rf1)
    # create_pca_test_graphs(res_rf10)
    # create_pca_test_graphs(res_rf20)
    # create_pca_test_graphs(res_rf40)
    # create_pca_test_graphs(res_svm1)
    # create_pca_test_graphs(res_svm10)
    # create_pca_test_graphs(res_svm20)
    # create_pca_test_graphs(res_svm40)
    # print(res_rf1)
    # print(res_rf10)
    # print(res_rf20)
    # print(res_rf40)
    # print(res_svm1)
    # print(res_svm10)
    # print(res_svm20)
    # print(res_svm40)
    #test_hyper_parameters(RFModel({'randomforestclassifier__n_estimators': [100,300,500], 'randomforestclassifier__max_depth': [50,100,300]}), path, 20, "center_norm", StandardScaler(), True, 116)
    #test_hyper_parameters(SvmModel({'svc__C': [math.pow(2, 3),  math.pow(2, 7),  math.pow(2, 11)], 'svc__gamma': [math.pow(2, 3),  math.pow(2, -3),  math.pow(2, -9), math.pow(2, -15)]}), path, 20, None, StandardScaler(), True, 20*18)

    #create_hyper_parameter_test_graph(load_hyper_parameter_test_results(SvmModel({})))
    #print(load_preprocessing_and_scaler_table_data(SvmModel({}), 20, "all"))
    #get_table(load_preprocessing_and_scaler_table_data(SvmModel({}), 20, "all"))
    # print(load_kernel_test_results())
    # print(get_table(load_kernel_test_results()))
    # print(find_best_result(SvmModel({}), 1))
    # print(find_best_result(RFModel({}), 1))
    # print(find_best_result(SvmModel({}), 10))
    # print(find_best_result(RFModel({}), 10))
    # print(find_best_result(SvmModel({}), 20))
    # print(find_best_result(RFModel({}), 20))
    # print(find_best_result(SvmModel({}), 40))
    # print(find_best_result(RFModel({}), 40))
    svm = SvmModel({'svc__C': [math.pow(2, 3), math.pow(2, 7), math.pow(2, 11)],
                    'svc__gamma': [math.pow(2, 3), math.pow(2, -3), math.pow(2, -9), math.pow(2, -15)]})
    rf = RFModel({'randomforestclassifier__n_estimators': [100, 300, 500], 'randomforestclassifier__max_depth': [50, 100, 300]})
    test_hyper_parameters(svm, path, 1, "center_norm", MinMaxScaler(), False, None)
    test_hyper_parameters(rf, path, 1, None, StandardScaler(), True, 1*18)
    test_hyper_parameters(svm, path, 10, "center_norm", MinMaxScaler(), False, None)
    test_hyper_parameters(rf, path, 10, None, MinMaxScaler(), True, 10*18)
    test_hyper_parameters(svm,path,20,None, StandardScaler(),True,20*18)
    test_hyper_parameters(rf,path, 20, "center_norm", StandardScaler(), True, 116)
    test_hyper_parameters(svm,path,40,None, MinMaxScaler(),True,40*18)
    test_hyper_parameters(rf, path, 40, None, StandardScaler(), True, 223)



