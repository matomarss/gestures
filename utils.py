import ctypes

import cv2
import numpy as np

finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']

TARGET_DICTS = {'point': 'p', 'grasp': 'g', 'move': 'm', 'ask': 'a', 'ok': 'o', 'nothing': 'x'}
LABELS_DICT = {'point': 0, 'grasp': 1, 'move': 2, 'ask': 3, 'ok': 4, 'nothing': 5}
INV_LABELS_DICT = {v: k for k, v in LABELS_DICT.items()}

from lib import Leap

def frame_list_from_dict(dictionary):
    data = dictionary['data']

    frame_list = []

    if 'Left Hand' in data:
        hand = data['Left Hand']
    else:
        hand = data['Right Hand']

    frame_list.extend(hand)

    for finger in finger_names:
        frame_list.extend(data['fingers'][finger])

    return frame_list


def convert_distortion_maps(image):
    distortion_length = image.distortion_width * image.distortion_height
    xmap = np.zeros(distortion_length // 2, dtype=np.float32)
    ymap = np.zeros(distortion_length // 2, dtype=np.float32)

    for i in range(0, distortion_length, 2):
        xmap[distortion_length // 2 - i // 2 - 1] = image.distortion[i] * image.width
        ymap[distortion_length // 2 - i // 2 - 1] = image.distortion[i + 1] * image.height

    xmap = np.reshape(xmap, (image.distortion_height, image.distortion_width // 2))
    ymap = np.reshape(ymap, (image.distortion_height, image.distortion_width // 2))

    # resize the distortion map to equal desired destination image size
    resized_xmap = cv2.resize(xmap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)
    resized_ymap = cv2.resize(ymap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)

    # Use faster fixed point maps
    coordinate_map, interpolation_coefficients = cv2.convertMaps(resized_xmap,
                                                                 resized_ymap,
                                                                 cv2.CV_32FC1,
                                                                 nninterpolation=False)

    return coordinate_map, interpolation_coefficients


def undistort(image, coordinate_map, coefficient_map, width, height):
    destination = np.empty((width, height), dtype=np.ubyte)

    # wrap image data in numpy array
    i_address = int(image.data_pointer)
    ctype_array_def = ctypes.c_ubyte * image.height * image.width
    # as ctypes array
    as_ctype_array = ctype_array_def.from_address(i_address)
    # as numpy array
    as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
    img = np.reshape(as_numpy_array, (image.height, image.width))

    # remap image to destination
    destination = cv2.remap(img,
                            coordinate_map,
                            coefficient_map,
                            interpolation=cv2.INTER_LINEAR)

    # resize output to desired destination size
    destination = cv2.resize(destination,
                             (width, height),
                             0, 0,
                             cv2.INTER_LINEAR)
    return destination


def process_frame(controller, show=False):
    '''Receives frame and video data, initiates recording on key press'''

    frame = controller.frame()
    image = frame.images[0]

    if image.is_valid:
        if show:
            left_coordinates, left_coefficients = convert_distortion_maps(frame.images[0])
            right_coordinates, right_coefficients = convert_distortion_maps(frame.images[1])
            maps_initialized = True

            undistorted_left = undistort(image, left_coordinates, left_coefficients, 400, 400)
            undistorted_right = undistort(image, right_coordinates, right_coefficients, 400, 400)

            # display images
            cv2.imshow('Left Camera', cv2.resize(undistorted_left, (1 * undistorted_left.shape[1], 1 * undistorted_left.shape[1])))
            cv2.imshow('Right Camera', cv2.resize(undistorted_right, (1 * undistorted_right.shape[1], 1 * undistorted_right.shape[1])))

            key = cv2.waitKey(1)

        '''Record hand and finger data from Leap Motion Controller'''
        finger_list = []
        frame_dict = {}

        for hand in frame.hands:
            hand_type = "Left Hand" if hand.is_left else "Right Hand"
            hand_coords = str(hand.palm_position)

            for pointable in hand.pointables:
                if pointable.is_finger:
                    # print("Receiving hand and finger data...")
                    finger = Leap.Finger(pointable)
                    finger_type = finger_names[finger.type]
                    finger_coords = str(finger.tip_position)
                    finger_coords = eval(finger_coords)
                    finger_list.append([finger_type, finger_coords])
                    frame_dict = {'timestamp': frame.timestamp,
                                  'data': {hand_type: eval(hand_coords), 'fingers': {k: v for k, v in finger_list}}}

            if len(frame_dict) > 0:
                print("Good recording")
        if show:
            return frame_dict, key, undistorted_right # decided to only return images of one hand
        else:
            return frame_dict

    if show:
        return None, None, None
    else:
        return {}