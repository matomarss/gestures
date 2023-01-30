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

from train_classifier import train_and_evaluate, parse_args, SvmModel, RFModel
from files_organization import dump_to_json, dump_object, load_from_json


def find_best_result(mod, n):
    best_result = {"validation_accuracy": 0}
    direc = "test_pca_n=" + str(n)
    filename_no_pca = mod.get_name() + "test_with_NO_PCA_n=" + str(n)
    filename_pca = mod.get_name() + "test_with_PCA_n=" + str(n)
    data = load_from_json(direc, filename_pca)
    for n_comp in data.keys():
        records = data[n_comp]
        for i in range(len(records)):
            val_acc = records[i].get("validation_accuracy")
            if val_acc > best_result.get("validation_accuracy"):
                best_result["validation_accuracy"] = val_acc
                best_result["n"] = n
                best_result["preprocessing"] = records[i].get("preprocessing")
                best_result["scaler"] = records[i].get("scaler")
                best_result["pca"] = n_comp
    data = load_from_json(direc, filename_no_pca)
    records = data.get("NO_PCA")
    for rec in records:
        val_acc = rec.get("validation_accuracy")
        if val_acc > best_result.get("validation_accuracy"):
            best_result["validation_accuracy"] = val_acc
            best_result["n"] = n
            best_result["preprocessing"] = rec.get("preprocessing")
            best_result["scaler"] = rec.get("scaler")
            best_result["pca"] = "None"
    return best_result


def load_pca_test_results_average(n, mod):
    accuracies = {}

    direc = "test_pca_n="+str(n)
    filename_no_pca = mod.get_name() + "test_with_NO_PCA_n="+str(n)
    filename_pca = mod.get_name() + "test_with_PCA_n="+str(n)

    data = load_from_json(direc, filename_pca)
    for n_comp in data.keys():
        if n_comp == "all":
            acc_key = n*18
        else:
            acc_key = int(n_comp)
        records = data[n_comp]
        sum_val_acc = 0
        for i in range(len(records)):
            sum_val_acc += records[i].get("validation_accuracy")
        val_acc = sum_val_acc/len(records)
        accuracies[acc_key] = val_acc

    data = load_from_json(direc, filename_no_pca)
    records = data.get("NO_PCA")
    sum_val_acc_no_scaler = 0
    sum_val_acc_scaler = 0
    for rec in records:
        if rec.get("scaler") == "None":
            sum_val_acc_no_scaler += rec.get("validation_accuracy")
        else:
            sum_val_acc_scaler += rec.get("validation_accuracy")
    val_acc_no_scaler = sum_val_acc_no_scaler/2
    val_acc_scaler = sum_val_acc_scaler/4
    accuracies[-2] = val_acc_no_scaler
    accuracies[-1] = val_acc_scaler

    return accuracies


def load_pca_best_test_results(n, mod):
    accuracies = {}

    direc = "test_pca_n=" + str(n)
    filename_no_pca = mod.get_name() + "test_with_NO_PCA_n=" + str(n)
    filename_pca = mod.get_name() + "test_with_PCA_n=" + str(n)
    data = load_from_json(direc, filename_pca)
    for n_comp in data.keys():
        if n_comp == "all":
            acc_key = n*18
        else:
            acc_key = int(n_comp)
        records = data[n_comp]
        accuracies[acc_key] = 0
        for i in range(len(records)):
            val_acc = records[i].get("validation_accuracy")
            if val_acc > accuracies.get(acc_key):
                accuracies[acc_key] = val_acc
    data = load_from_json(direc, filename_no_pca)
    records = data.get("NO_PCA")
    accuracies[-1] = 0
    for rec in records:
        val_acc = rec.get("validation_accuracy")
        if val_acc > accuracies.get(-1):
            accuracies[-1] = val_acc

    return accuracies


def load_pca_test_results(n, mod):
    final = {}

    direc = "test_pca_n=" + str(n)
    filename_no_pca = mod.get_name() + "test_with_NO_PCA_n=" + str(n)
    filename_pca = mod.get_name() + "test_with_PCA_n=" + str(n)
    data = load_from_json(direc, filename_pca)
    final[n*18] = data.get("all")
    for n_comp in data.keys():
        if n_comp == "all":
            continue
        final[int(n_comp)] = data.get(n_comp)

    data = load_from_json(direc, filename_no_pca)
    final[-1] = data.get("NO_PCA")

    final = dict(sorted(final.items()))

    return final


def load_kernel_test_results_for_pca_table(pca):
    results = load_from_json("test_svm_kernels", "SVM_kernel_test_n=20")
    processed_results = {}
    for kernel in results.keys():
        processed_results[kernel] = {}
        for rec in results.get(kernel):
            if rec.get("pca") != pca:
                continue
            compound = ""
            for key in rec.keys():
                if key != "validation_accuracy":
                    compound += "&"+str(rec.get(key))
            processed_results[kernel][compound] = rec["validation_accuracy"]
    return processed_results


def load_hyper_parameter_test_results(mod):
    results = {}
    direc = "test_hyper_parameters"
    for n in [1, 10, 20, 40]:
        filename = mod.get_name() + "_hyper_par_test_n=" + str(n)
        data = load_from_json(direc, filename)
        results[n] = data.get("best_parameters").get("best_score")
    return results


def load_preprocessing_and_scaler_table_data(mod, n, pca):
    res = load_pca_test_results(n, mod)

    if pca == "all":
        pca = n*18
    if pca is None:
        pca = -1
    ld = {}
    for prep in ["None", "center_norm"]:
        ld[prep] = {}
        for rec in res.get(pca):
            if prep == rec.get("preprocessing"):
                ld[prep][rec.get("scaler")] = rec.get("validation_accuracy")
    return ld
