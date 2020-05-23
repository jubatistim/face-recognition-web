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
    try:
        #cv2 image to string base 64 encoded
        print('a000')
        _, buffer = cv2.imencode('.jpg', img_return)
        print('a001')
        image_read = base64.b64encode(buffer)
        print('a002')
        # string base 64 encoded to cv2 image - THIS CHANGES COLOR OF THE IMAGE AND THE RECOGNITION LOOKS BETTER - I DON'T KNOW WHY IT CHANGES IMAGE COLOR
        decoded = base64.b64decode(image_read)
        print('a003')
        nimage = Image.open(io.BytesIO(decoded))
        print('a004')
        img_source = np.array(nimage)
        print('a005')

        # pre processing
        imgSource = img_source.copy()
        print('a006')
        imgReturn = img_return.copy()
        print('a007')

        if imgReturn.shape[1] != imgSource.shape[1] or imgReturn.shape[0] != imgSource.shape[0]:
            raise Exception("The shapes of img_source and img_return should be the same.")
            print('a008')

        if imgReturn.shape[1] > imgReturn.shape[0]:
            print('a009')
            if imgReturn.shape[1] > max_wh:
                print('a010')
                n_width = max_wh
                print('a011')
                n_height = max_wh * imgReturn.shape[0] / imgReturn.shape[1]
                print('a012')

                imgReturn = cv2.resize(imgReturn, (int(n_width), int(n_height)), interpolation = cv2.INTER_AREA)
                print('a013')
                imgSource = cv2.resize(imgSource, (int(n_width), int(n_height)), interpolation = cv2.INTER_AREA)
                print('a014')
            else:
                min_size = 32
                print('a015')
                face_padding = 30
                print('a016')
        else:
            if imgReturn.shape[0] > max_wh:
                print('a017')
                n_height = max_wh
                print('a018')
                n_width = max_wh * imgReturn.shape[1] / imgReturn.shape[0]
                print('a019')

                imgReturn = cv2.resize(imgReturn, (int(n_width), int(n_height)), interpolation = cv2.INTER_AREA)
                print('a020')
                imgSource = cv2.resize(imgSource, (int(n_width), int(n_height)), interpolation = cv2.INTER_AREA)
                print('a021')
            else:
                min_size = 32
                print('a022')
                face_padding = 30  
                print('a023')  
        
        gray = cv2.cvtColor(imgSource, cv2.COLOR_BGR2GRAY)
        print('a024')
        faces = face_cascade.detectMultiScale(gray, 1.10, 8)
        print('a025')

        # save detected faces
        for (x, y, w, h) in faces:
            print('a026')

            if w > min_size and h > min_size:

                print('a027')
                crop_img = imgSource[y-face_padding:y+h+face_padding, x-face_padding:x+w+face_padding]

                print('a028')
                crop_img = cv2.resize(crop_img, (64, 64), interpolation = cv2.INTER_AREA)

                print('a029')
                test_image = image_utils.img_to_array(crop_img)
                print('a030')
                test_image = np.expand_dims(test_image, axis = 0)
                
                # validate
                print('a031')
                global sess
                print('a032')
                global graph
                print('a033')
                with graph.as_default():
                    print('a034')
                    set_session(sess)
                    print('a035')
                    result = model.predict_on_batch(test_image)

                print('a036')
                who = ''

                print('a037')
                if result[0][0] == 0:
                    print('a038')
                    who = 'JULIANO'
                    print('a039')
                    cv2.rectangle(imgReturn, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    print('a040')
                    cv2.putText(imgReturn, who, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    print('a041')
                    who = 'LUNA'
                    print('a042')
                    cv2.rectangle(imgReturn, (x, y), (x+w, y+h), (191, 0, 255), 2)
                    print('a043')
                    cv2.putText(imgReturn, who, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (191, 0, 255), 2)
        print('a044')

        return imgReturn
    except:
        print('Deu merda')