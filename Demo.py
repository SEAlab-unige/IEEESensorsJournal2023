import numpy as np
import os
import cv2
import math
from tensorflow.keras.models import load_model
import tensorflow as tf

def get_image_array(img, width, height):
    img = img.astype(np.float32)
    img = image_resize_no_dist(img, width, height)
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    return img

def image_resize_no_dist(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]
    if (w > h):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None if width is None: calculate the ratio of the height and construct the dimensions
    r = height / float(h)
    if (int(w * r) < width):
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    (h, w) = resized.shape[:2]

    delta_w = width - w
    delta_h = height - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0,0,0]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) #BORDER_REPLICATE)#
    return padded


""" MAIN """
test_images_path = "./ExampleInputs/"
reconstructed_model = load_model("./Models/NAS_UMD_0.25s")

out_shape = int(math.sqrt(reconstructed_model.outputs[0].shape[1]))
in_shape =  int(reconstructed_model.inputs[0].shape[1])

images_path = os.listdir(test_images_path)
for index, path in enumerate(images_path):
    print(test_images_path + path)

    im = cv2.imread(test_images_path+ path, 1)
    input = get_image_array(im, in_shape, in_shape)

    P = reconstructed_model.predict(np.expand_dims(input,axis =0))
    P = np.squeeze(P[0])
    Pi = np.argmax(P, axis=1)
    prediction_raw = Pi.reshape((out_shape, out_shape))

    black_raw = np.zeros((out_shape, out_shape, 3))
    black_raw[prediction_raw ==1] = [255, 0, 0]  # 'grasp' -> blue
    black_raw[prediction_raw ==2] = [0, 255, 0]  # 'don't touch' _> green

    disp_input = image_resize_no_dist(im, in_shape, in_shape)
    cv2.imshow('Image', cv2.resize(disp_input, (672, 672)))
    cv2.imshow('Prediction', cv2.resize(black_raw, (672, 672)))

    key = cv2.waitKey(0)


