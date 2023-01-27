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


def dump_to_json(path, filename, data):
    path = os.path.join(path, filename) + ".json"
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_from_json(path, filename):
    path = os.path.join(path, filename) + ".json"
    with open(path, "r") as f:
        data = json.load(f)
    return data


def dump_object(path, filename, obj):
    path = os.path.join(path, filename) + ".joblib"

    joblib.dump(obj, path)
