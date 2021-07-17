import io

import flask
import numpy as np
from keras.applications import imagenet_utils
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import img_to_array
from PIL import Image

app = flask.Flask(__name__)
model = None


def load_model():
    """
    load the pre-trained Keras model(here we are using a model
    pretreined on ImageNet and provided by Keras, but you can
    substitute in your own networks just as easily)
    """
    global model
    model = ResNet50(weights="imagenet")


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
        preds = model.predict(image)
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
    load_model()
    app.run()
