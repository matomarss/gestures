import keras
import os
import cv2
import numpy as np

import loader

if __name__ == '__main__':
    model_path = 'snapshots/gestures_03-0.66.hdf5'
    model = keras.models.load_model(model_path)

    dataset_path = 'D:/Skola/PhD/data/gesture_dataset_2018_09_18/dataset/'

    image_names = ['getr1_11/right12.jpg', 'putr1_12/right17.jpg', 'palmr_12/right10.jpg', 'paol_11/left20.jpg']
    orig_images = []
    input_images = []

    for name in image_names:
        image_path = os.path.join(dataset_path, name)
        orig_images.append(cv2.imread(image_path))
        input_images.append(loader.load_image(image_path))

    x = np.stack(input_images)

    predictions = model.predict(x, batch_size=None, verbose=0, steps=None)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cls_dict = {v: k for k, v in loader.CLS_DICT.items()}
    for i, image in enumerate(orig_images):
        best = np.argmax(predictions[i])
        conf = predictions[i,best]
        cls = cls_dict[best]
        image = cv2.putText(image, '{}:{}'.format(cls, conf), (50,50), font, 1,(0,0,255))
        cv2.imshow("Image", image)
        cv2.waitKey(0)

