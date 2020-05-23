from flask import Flask, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy
import calendar
import time
from custom_util import *

app=Flask(__name__)

# get running path
base_dir = os.path.dirname(__file__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/success", methods=['POST'])
def success():
    if request.method=='POST':
        filestr=request.files["file"]

        print('000')

        #convert string data to numpy array
        npimg = numpy.fromstring(filestr.read(), numpy.uint8)
        print('001')
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.COLOR_RGB2BGR)
        print('002')
        
        image_predicted = predict_Luna_Ju(img)
        print('003')
            
        file_to_save = str(calendar.timegm(time.gmtime()))
        print('004')
        cv2.imwrite(os.path.join(base_dir, 'static', file_to_save + '.jpg'), image_predicted)
        print('005')
        image_file = url_for('static', filename=file_to_save + '.jpg')

        print('AAA')

        return render_template("success.html", img = image_file)

        print('BBB')

if __name__ == '__main__':
    app.run()