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
    if os.path.exists(path):
        with open(path, "a") as f:
            json.dump(data, f, indent=4)
    else:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)


def dump_object(path, filename, obj):
    path = os.path.join(path, filename) + ".joblib"

    joblib.dump(obj, path)
