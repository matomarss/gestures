"""
This program reads the raw data JSON files, flattens them in numpy array format,
matches the label in one-hot-encoding and stores the features and targets for each
gesture per participant in .npz format

1. Get help: to get more explanation about the required arguments pass the following in your command line:
python dataPrep.py -h

2. Run the script:
run the following in your command line by replacing <...> with your information:
python dataPrep.py <'input_path'> <'output_path'> <'participant'>

note: 'input_path' has to contain the files of one participant (equal to the participant parameter)

optional:
if you want to save normalized data over every feature individually run:
python dataPrep.py <'input_path'> <'output_path'> <'participant'> --normalize

if you want to save standardized data over every feature individually run:
python dataPrep.py <'input_path'> <'output_path'> <'participant'> --standardize

if you use different gestures from the one's used in our experiment, please change the target_dict manually

python dataPrep.py '/Users/claraswaboda/Desktop/CogSci/Semester_3/4_Project/3_data/esn_data_processing/data_recording_2/JSON_recording_2/both_hands/janci/*' '/Users/claraswaboda/Desktop/CogSci/Semester_3/4_Project/3_data/esn_data_processing/data_recording_2/noNothing_normalized_whole_space/both_hands/individual_files/' 'janci'

author: Clara <3
"""

import os, argparse
import json
import numpy as np

from utils import LABELS_DICT, frame_list_from_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description='flatten JSON data file, match targets, store features & targets in .npz format')
    parser.add_argument('root_path', help='path to the individual data files in .npz format')
    parser.add_argument('output_path', help='path to the output folder for the data files')
    args = parser.parse_args()
    return args


def flatten_features(features):
    '''extract recording data from JSON file and flattens the features into a list of
    feature arrays, extract dictionary keys as column names for data matrix'''

    feature_array = []

    for dictionary in features:
        frame_list = frame_list_from_dict(dictionary)
        feature_array.append(frame_list)

    return np.array(feature_array)


def get_features_targets(filename, json_data):
    '''extract label from filename and assign labels to data (one-hot-encoding) '''

    flat_features = flatten_features(json_data)

    basename = os.path.basename(filename)
    if basename[-5:] == '.json':
        record_id = basename[:-5].split('_')[1]
        basename = basename[:-5].split('_')[0]

    targets = np.zeros([flat_features.shape[0], 6])
    targets[:, LABELS_DICT[basename]] = 1

    return flat_features, targets


def load_data(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
        features, targets = get_features_targets(json_path, json_data)

    return features, targets


def parse_files(path, out_path):
    participants = [f for f in os.listdir(path)]

    for participant in participants:
        participant_dir_path = os.path.join(path, participant)
        gesture_json_filenames = [f for f in os.listdir(participant_dir_path) if '.json' in f]
        for gesture_json_filename in gesture_json_filenames:
            gesture_json_path = os.path.join(participant_dir_path, gesture_json_filename)
            gesture_npz_path = os.path.join(out_path, participant, '{}.npz'.format(gesture_json_filename.split('.')[0]))

            features, targets = load_data(gesture_json_path)

            if not os.path.exists(os.path.dirname(gesture_npz_path)):
                os.makedirs(os.path.dirname(gesture_npz_path))

            np.savez(gesture_npz_path, features=features, targets=targets)


if __name__ == "__main__":
    args = parse_args()
    parse_files(args.root_path, args.output_path)
