import os
import io
import json
import base64
import hashlib
import logging
import logging.config
import boto3
import tempfile
import numpy as np

from urllib.parse import urlparse
from flask import Flask, request, Response

from PIL import Image

from model import Model


PREFIX_PATH = "/opt/ml/"
CACHE_PATH = os.path.join(PREFIX_PATH, "cache")
MODEL_PATH = os.path.join(PREFIX_PATH, "model")
FROZEN_GRAPH_PATH = os.path.join(MODEL_PATH, "frozen_inference_graph.pb")
LABEL_PATH = os.path.join(MODEL_PATH, "label_map.pbtxt")
PARAM_PATH = os.path.join(MODEL_PATH, "hyperparameters.json")


class Configuration:
    def __init__(self, data: dict):
        self.file = data.get("file", None)
        self.image = data.get("image", None)
        self.stride = data.get("stride", 1)

        self.cache = data.get("cache", False)
        self.cache_id = data.get("cache_id", None)

        self.aws_region = data.get("aws_region", None)
        self.aws_access_key = data.get("aws_access_key", None)
        self.aws_secret_access_key = data.get("aws_secret_access_key", None)


class Cache:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.enabled = configuration.cache
        self.path = CACHE_PATH

    def get(self, key: str):
        if self.enabled:
            entry = os.path.join(self.path, f"{key}.npz")
            if os.path.exists(entry):
                data = np.load(entry, allow_pickle=True)
                return data["data"]

        return None

    def put(self, key: str, value):
        if self.enabled:
            np.savez_compressed(os.path.join(self.path, f"{key}.npz"), data=value)


class Processor:
    model = None
    training_params = None

    @staticmethod
    def factory(configuration: Configuration):
        if configuration.image is not None or (
            configuration.file is not None
            and configuration.file[len(configuration.file) - 4 :] in (".jpg", ".png",)
        ):
            return ImageProcessor(configuration, Cache(configuration))

        return None

    @classmethod
    def get_training_params(self):
        if self.training_params is None:
            with open(PARAM_PATH, "r") as tc:
                self.training_params = json.load(tc)

        return self.training_params

    @classmethod
    def predict(self, image):
        return self.get_model().inference(image)

    @classmethod
    def get_model(self):
        if self.model is None:
            self.model = Model(LABEL_PATH, FROZEN_GRAPH_PATH)

        return self.model

    def __init__(self, configuration: Configuration, cache: Cache):
        self.configuration = configuration
        self.cache = cache

    def inference(self):
        # This method should be implemented on the sub-classes.
        pass


class ImageProcessor(Processor):
    def __init__(self, configuration: Configuration, cache: Cache):
        super(ImageProcessor, self).__init__(configuration, cache)

    def inference(self):
        logging.info(f"Running inference on image...")
        inference_cache_key = self.__get_key(self.__get_source(), "inference")

        predictions = None

        value = self.cache.get(inference_cache_key)
        if value is not None:
            predictions = value.tolist()
            logging.debug(f"Inference from image found in cache")

        if predictions is None:
            logging.debug(f"Inference from image not found in cache")
            image = self.__get_image()
            inference = Processor.predict(image)

            predictions = {"predictions": []}

            for index, label in enumerate(inference["detection_classes"]):
                box = inference["detection_boxes"][index].astype(float)
                score = float(inference["detection_scores"][index])

                ymin = box[0]
                xmin = box[1]
                ymax = box[2]
                xmax = box[3]

                prediction = [
                    float(label - 1),
                    score,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ]

                predictions["predictions"].append(prediction)

            self.cache.put(inference_cache_key, np.array(predictions))

        return predictions

    def __get_source(self):
        if self.configuration.image is not None:
            return base64.b64decode(self.configuration.image)

        return self.configuration.file

    def __get_image(self):
        source_cache_key = self.__get_key(self.__get_source(), "source")

        image = self.cache.get(source_cache_key)

        if image is None:
            logging.debug(f"Image not found in cache")

            try:
                if self.configuration.image is not None:
                    image = self.__get_image_from_base64_string(self.__get_source())
                else:
                    fragments = urlparse(self.__get_source(), allow_fragments=False)
                    if fragments.scheme == "s3":
                        image = self.__get_image_from_s3(self.__get_source(), fragments)

                self.cache.put(source_cache_key, image)
            except Exception as e:
                logging.error("There was an error handling image", e)
        else:
            logging.debug(f"Image found in cache")

        return image

    def __get_image_from_base64_string(self, source) -> np.array:
        image = Image.open(io.BytesIO(source))
        return self.__numpy(image)

    def __get_image_from_s3(self, file, fragments) -> np.array:
        logging.info(f"Downloading image from S3. Filename {file}...")

        s3 = (
            boto3.client(
                "s3",
                region_name=self.configuration.aws_region,
                aws_access_key_id=self.configuration.aws_access_key,
                aws_secret_access_key=self.configuration.aws_secret_access_key,
            )
            if self.configuration.aws_access_key
            else boto3.client("s3")
        )

        tmp = tempfile.NamedTemporaryFile()

        with open(tmp.name, "wb") as f:
            s3.download_fileobj(fragments.netloc, fragments.path[1:], f)
            return self.__numpy(Image.open(tmp.name))

    def __get_key(self, file: str, suffix: str):
        tmp = hashlib.md5(str(file).encode("utf8"))
        tmp.update(str(suffix).encode("utf8"))
        return tmp.hexdigest()

    def __numpy(self, image):
        (width, height) = image.size
        return np.array(image.getdata()).reshape((height, width, 3)).astype(np.uint8)


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# gunicorn_error_logger = logging.getLogger('gunicorn.error')
# app.logger.handlers.extend(gunicorn_error_logger.handlers)
# app.logger.setLevel(logging.DEBUG)


@app.route("/ping", methods=["GET"])
def ping():
    """This endpoint determines whether the container is working and healthy.
    We know everything is working appropriately if the model can be loaded
    successfully.
    """
    logging.info("Ping received...")

    health = Processor.get_model() is not None

    status = 200 if health else 404
    return Response(response="\n", status=status, mimetype="application/json")


@app.route("/deleteme", methods=["POST"])
def delete_me():
    """
    logging.info("Invoked with content_type {}".format(request.content_type))

    if request.content_type == "application/json":
        logging.info("Running inference on image...")

        body = request.get_json()
        image_data = base64.b64decode(body["image"])

        if "threshold" in body:
            threshold = body["threshold"]
        else:
            threshold = 0.0

        training_params = ScoringService.get_training_params()

        image_size = (
            int(training_params["image_size"])
            if "image_size" in training_params
            else 300
        )

        # run inference on image
        inference_result = ScoringService.predict(image_data, image_size)

        # convert inference result to json
        prediction_objects = []
        predictions = {}
        detection_classes = inference_result["detection_classes"]
        detection_boxes = inference_result["detection_boxes"]
        detection_scores = inference_result["detection_scores"]
        for index, detection_class in enumerate(detection_classes):
            detection_box = detection_boxes[index].astype(float)
            detection_score = float(detection_scores[index])
            if detection_score < threshold:
                continue

            ymin = detection_box[0]
            xmin = detection_box[1]
            ymax = detection_box[2]
            xmax = detection_box[3]

            prediction_object = [
                float(detection_class - 1),
                detection_score,
                xmin,
                ymin,
                xmax,
                ymax,
            ]

            prediction_objects.append(prediction_object)

        predictions["prediction"] = prediction_objects

        return Response(
            response=json.dumps(predictions), status=200, mimetype="application/json"
        )

    return Response(
        response='{"reason" : "Request is not application/x-image"}',
        status=400,
        mimetype="application/json",
    )
    """
    pass


@app.route("/invocations", methods=["POST"])
def invoke():
    """
    TODO:

    * Add support to provide http image
    * Add support to provide local image (file instead of S3)
    
    * Add support to provide a threshold.

    * Add support to provide a video: file | stride

    * Implement gRPC interface

    * What happens if there are no detections?
    * Is the "return only these classes" working?
    * What should we do with the hyperparameters.json?

    * Add support to use different frozen models.


    """

    if request.content_type == "application/json":
        configuration = Configuration(request.get_json())
        processor = Processor.factory(configuration)
        result = processor.inference()

        logging.debug(f"Inference result {result}")

        return Response(
            response=json.dumps(result), status=200, mimetype="application/json",
        )

    return Response(
        response='{"reason" : "Request is not application/x-image"}',
        status=400,
        mimetype="application/json",
    )
