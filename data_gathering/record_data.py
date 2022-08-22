import sys, os, inspect, time

from utils import process_frame

src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
arch_dir = '../lib/x64' if sys.maxsize > 2 ** 32 else '../lib/x86'
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, '../lib')))

# set recording directory path here
#record_path = 'C:\Clara\LeapScripts\Clara'
record_path = 'C:\Program Files (x86)\Leap Motion\LeapSDK\samples\data'

import Leap, json, cv2


# create dictionary of keys for each action (action types + stop)

action_dict = {ord('p'): 'point',
               ord('g'): 'grasp',
               ord('m'): 'move',
               ord('a'): 'ask',
               ord('o'): 'ok',
               ord('x'): 'nothing'
               }

def save(dict_list, frame_list, record_id, action_type, record_path=record_path):
    ''' save in user directory (each user has own directory), user directory has to be manually set
    at the beginning of the program
    file name: action type_recordingID '''

    print("Saving")

    if not os.path.isdir(record_path):
        os.mkdir(record_path)

    out_path = os.path.join(record_path, str(action_type) + "_" + str(record_id))

    with open(os.path.join(out_path + ".json"), "w") as f:
        json.dump(dict_list, f, indent=4, separators=(',', ': '))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(os.path.join(out_path + ".mp4"), fourcc, 10.0, (400, 400), isColor=False)

    # pass the list of frames here
    for image in frame_list:
        out.write(image)
    out.release()


def main():
    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

    record_id = 0
    dict_list, frame_list = [], []

    start_time = time.time()

    while True:
        frame_dict, pressed_key, images = process_frame(controller)

        if frame_dict is None:
            continue

        dict_list.append(frame_dict)
        frame_list.append(images)

        if pressed_key in action_dict.keys():
            # saves the sequence as given gesture
            save(dict_list, frame_list, record_id, action_dict[pressed_key])
            record_id += 1
            dict_list, frame_list = [], []
        elif pressed_key == ord('s'):
            # starts new sequence
            dict_list, frame_list = [], []
        elif pressed_key == ord('q'):
            # ends the annotator
            print("Application stopped manually")
            break
        print("FPS: " + str(1 / (start_time - time.time())))
        start_time = time.time()

if __name__ == "__main__":
    main()