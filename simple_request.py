import requests

# initialize the Keras REST API endpoint URL along with
# the input image path
KERAS_REST_API_URL = "http://localhost:5000/api/predict"
IMAGE_PATH = "data/cougar.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensures the request was sucessful
if r["success"]:
    # loop over the predictions and display them
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(
            i + 1, result["label"], result["probabily"]))
else:
    print("Request failed")
