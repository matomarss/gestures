import os
import joblib
import json


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