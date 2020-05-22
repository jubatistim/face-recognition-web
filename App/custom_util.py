import cv2
import os
import calendar
import time
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io
from keras.preprocessing import image as image_utils
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

# load model
sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)

model = load_model('./saved-models/cnn1589854703.h5')

# Face reconition classifier https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

def predict_Luna_Ju(img_return, max_wh = 700, min_size = 32, face_padding = 30):

    #cv2 image to string base 64 encoded
    _, buffer = cv2.imencode('.jpg', img_return)
    image_read = base64.b64encode(buffer)

    # string base 64 encoded to cv2 image - THIS CHANGES COLOR OF THE IMAGE AND THE RECOGNITION LOOKS BETTER - I DON'T KNOW WHY IT CHANGES IMAGE COLOR
    decoded = base64.b64decode(image_read)
    nimage = Image.open(io.BytesIO(decoded))
    img_source = np.array(nimage)

    # pre processing
    imgSource = img_source.copy()
    imgReturn = img_return.copy()

    if imgReturn.shape[1] != imgSource.shape[1] or imgReturn.shape[0] != imgSource.shape[0]:
        raise Exception("The shapes of img_source and img_return should be the same.")

    if imgReturn.shape[1] > imgReturn.shape[0]:
        if imgReturn.shape[1] > max_wh:
            n_width = max_wh
            n_height = max_wh * imgReturn.shape[0] / imgReturn.shape[1]

            imgReturn = cv2.resize(imgReturn, (int(n_width), int(n_height)), interpolation = cv2.INTER_AREA)
            imgSource = cv2.resize(imgSource, (int(n_width), int(n_height)), interpolation = cv2.INTER_AREA)
        else:
            min_size = 32
            face_padding = 30
    else:
        if imgReturn.shape[0] > max_wh:
            n_height = max_wh
            n_width = max_wh * imgReturn.shape[1] / imgReturn.shape[0]

            imgReturn = cv2.resize(imgReturn, (int(n_width), int(n_height)), interpolation = cv2.INTER_AREA)
            imgSource = cv2.resize(imgSource, (int(n_width), int(n_height)), interpolation = cv2.INTER_AREA)
        else:
            min_size = 32
            face_padding = 30    
    
    gray = cv2.cvtColor(imgSource, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.10, 8)

    # save detected faces
    for (x, y, w, h) in faces:

        if w > min_size and h > min_size:

            crop_img = imgSource[y-face_padding:y+h+face_padding, x-face_padding:x+w+face_padding]

            crop_img = cv2.resize(crop_img, (64, 64), interpolation = cv2.INTER_AREA)

            test_image = image_utils.img_to_array(crop_img)
            test_image = np.expand_dims(test_image, axis = 0)
            
            # validate
            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                result = model.predict_on_batch(test_image)

            who = ''

            if result[0][0] == 0:
                who = 'JULIANO'
                cv2.rectangle(imgReturn, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(imgReturn, who, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                who = 'LUNA'
                cv2.rectangle(imgReturn, (x, y), (x+w, y+h), (191, 0, 255), 2)
                cv2.putText(imgReturn, who, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (191, 0, 255), 2)

    return imgReturn