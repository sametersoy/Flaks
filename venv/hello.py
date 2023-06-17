# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, request, jsonify
from teachable_machine import TeachableMachine
import cv2 as cv
import requests
import json
from flask_cors import CORS
import base64
import numpy as np
import io



model = TeachableMachine(model_path="converted_keras\keras_model.h5",
                         labels_file_path="converted_keras\labels.txt")

image_path = "converted_keras\screenshot.jpg"
 
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
CORS(app, supports_credentials=True) 


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world(metadata):
    print(metadata)
    URL = "https://teachablemachine.withgoogle.com/models/nuEIHI9PQ/"
    modelURL = URL + "model.json"
    metadataURL = URL + "metadata.json"

    result = model.classify_image(image_path)
    class_label = result["class_name"]
    return class_label

@app.route('/samet', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Kameradan resim yakala
        image = capture_image()

        # Yakalanan resmi diske kaydet
        cv.imwrite('uploaded_image.jpg', image)

        return 'Resim başarıyla yüklendi!'
    
@app.route('/process-snapshot', methods=['POST'])
def process_snapshot():

    data = request.get_json()

    # Extract the image data from the data URL
    image_data = base64.b64decode(data['image'].split(",")[1])
    #print(image_data)
    # Convert the image data to an OpenCV image (numpy array)
    image = cv.imdecode(np.frombuffer(image_data, np.uint8), -1)
    
    cv.imwrite('converted_keras/screenshot.jpg', image)

    result = model.classify_image(image_path)
    print(result)
    predictions = result["predictions"]
    print(type(predictions))
    data = np.array(predictions)
    print(type(data))
    new = data[:, 1]
    print(predictions[0,1])
    p=predictions[0,1]
    class_label = result["class_name"]
    x={"data":class_label,"predictions":str(p)}
    y = json.dumps(x)
    print(type(y))


    return y
# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()