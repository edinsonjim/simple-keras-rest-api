import io

import flask
import numpy as np
import tensorflow as tf
from keras.applications import imagenet_utils
from keras.applications.resnet import ResNet50
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow import keras

app = flask.Flask(__name__)
model = None
rest_net_model = None
italian_docs_net_model = None


def load_rest_net_model():
    global rest_net_model
    rest_net_model = ResNet50(weights="imagenet")


def load_italian_docs_net_model():
    """
    load the pre-trained Keras model(here we are using a model
    pretreined on ImageNet and provided by Keras, but you can
    substitute in your own networks just as easily)
    """
    global italian_docs_net_model
    italian_docs_net_model = load_model("models/italian_docs_net.h5")


def prepare_image(image, target):
    """
    Preprocess the image and prepare it for classification

    * Converts the mode to RGB(if necessary)
    * Resizes it to 224x225 pixeles(the input spatial dimensions for RestNet)
    * Preprocessess the array via mean substraction and scaling
    """
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


def prepare_image_to_italian_docs(image, target):
    """
    Preprocess the image and prepare it for classification

    * Resizes it to 180x180 pixeles(the input spatial dimension for italian docs net)
    """
    image = image.resize(target)
    image = img_to_array(image)
    image = tf.expand_dims(image, 0)

    return image


@app.route("/api/italian-docs/predict", methods=["POST"])
def italian_docs_predict():
    data = {"success": False}

    if flask.request.files.get("image"):
        # read the image in PIL format
        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))

        # preprocess the image and prepare it for classification
        image = prepare_image_to_italian_docs(image, target=(180, 180))

        # clasify the input image and then initialize the list
        # of predictions to return to the client
        predictions = italian_docs_net_model.predict(image)

        class_names = ['Carta di Identita',
                       'Codice Fiscale', 'Carta di Soggiorno']

        score = tf.nn.softmax(predictions[0])

        data["predictions"] = []
        result = {"label": class_names[np.argmax(score)],
                  "probabily": 100 * np.max(score)}
        data["predictions"].append(result)

        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Initialize the data directory that will be returned from the view
    """
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.files.get("image"):
        # read the image in PIL format
        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))

        # preprocess the image and prepare it for classification
        image = prepare_image(image, target=(224, 224))

        # clasify the input image and then initialize the list
        # of predictions to return to the client
        preds = rest_net_model.predict(image)
        resutls = imagenet_utils.decode_predictions(preds)
        data["predictions"] = []

        # loop over the results and add them to the list of
        # returned predictions
        for (_, label, prob) in resutls[0]:
            result = {"label": label, "probabily": float(prob)}
            data["predictions"].append(result)

        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_rest_net_model()
    load_italian_docs_net_model()

    app.run()
