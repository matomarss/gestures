import argparse
import sys

import cv2
import joblib
import numpy as np

from utils import undistort, convert_distortion_maps, finger_names, process_frame, frame_list_from_dict, INV_LABELS_DICT

sys.path.insert(0, "./lib")
from lib import Leap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_frames', type=int, default=10, help='How many frames to use')
    args = parser.parse_args()
    return args


def main(n):
    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

    feature_array = np.zeros([n, 18])

    model_path = 'models/RF_n{}.joblib'.format(n)
    pipeline = joblib.load(model_path)

    while True:
        dictionary = process_frame(controller)
        if len(dictionary) > 0:
            frame_list = frame_list_from_dict(dictionary)

            if n == 1:
                feature_array = np.array(frame_list)
            else:
                feature_array[:n - 1] = feature_array[1:]
                feature_array[n - 1] = np.array(frame_list)

            prediction = pipeline.predict_proba(feature_array.reshape(1, -1))
            print("Probabilities: ", " ".join(["{}: {:04f}, ".format(INV_LABELS_DICT[i], j) for i, j in enumerate(prediction[0])]))
            print("Best gesture:", INV_LABELS_DICT[np.argmax(prediction[0])])

if __name__ == '__main__':
    args = parse_args()
    main(args.num_frames)
