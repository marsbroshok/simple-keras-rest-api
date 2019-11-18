# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.models import load_model as keras_load_model
from keras.preprocessing.image import img_to_array, random_zoom
from keras.applications import imagenet_utils
from keras.backend import set_session
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
session = None
model = None
idx_to_class = None



def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    # load as well the dictionary with labels to match with predictions
    global session
    session = tf.Session()
    set_session(session)

    global model
    model_filepath = "models/export-full-model.hdf5"
    model = keras_load_model(model_filepath)
    model._make_predict_function()

    global idx_to_class
    with open("models/classes.json") as f:
        idx_to_class = json.load(f)


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = random_zoom(image, (0.5, 0.5), 0, 1, 2)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

@app.route('/')
def index():
    # html_file = 'view/index.html'
    return flask.send_from_directory(directory='view', filename='index.html')

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            # flask.request.form()["file"]
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(197, 197))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            with session.as_default():
                with session.graph.as_default():
                    preds = model.predict(image)[0]
            top = 5
            top_indices = preds.argsort()[-top:][::-1]

            results = [(idx_to_class[str(i)], str(preds[i])) for i in top_indices]
            data["predictions"] = results

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(debug=True, host='0.0.0.0')
