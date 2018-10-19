import os
import cv2
import keras
import numpy as np

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

CLS_DICT = {'put': 0 , 'mov': 1, 'ofo': 2, 'ono': 3, 'palm': 4, 'pao': 5, 'get': 6}

def get_class_number(folder):
    name = os.path.basename(folder)
    possible = [x for x in CLS_DICT.keys() if x in name]
    if len(possible) > 0:
        return CLS_DICT[possible[0]]
    else:
        return None



def load_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # img = cv2.imread(path)
    # img = cv2.resize(img, (224, 224))
    # x = img.astype(keras.backend.floatx())
    # x[..., 0] -= 103.939
    # x[..., 1] -= 116.779
    # x[..., 2] -= 123.68
    return x[0]


def load(dataset_path, list, limit):
    X = []
    Y = []
    subfolders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]

    for folder in subfolders:
        if int(folder.split('_')[-1]) not in list:
            continue
        cls = get_class_number(folder)
        if cls is None:
            continue
        n_files = int(len([name for name in os.listdir(folder)])/2)
        for i in range(limit,n_files-limit):
            r_name = 'right{}.jpg'.format(i+1)
            l_name = 'left{}.jpg'.format(i+1)
            r_img = load_image(os.path.join(folder, r_name))
            l_img = load_image(os.path.join(folder, l_name))
            X.append(r_img)
            Y.append(cls)
            X.append(l_img)
            Y.append(cls)
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    Y = keras.utils.np_utils.to_categorical(Y)
    return X,Y