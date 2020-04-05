import os
import io
import json
import base64
import hashlib
import logging
import logging.config
import boto3
import tempfile
import requests
import numpy as np

from urllib.parse import urlparse
from flask import Flask, request, Response

from PIL import Image
from PIL import ImageFile

from model import Model


ImageFile.LOAD_TRUNCATED_IMAGES = True

PREFIX_PATH = "/opt/ml/"
CACHE_PATH = os.path.join(PREFIX_PATH, "cache")
MODEL_PATH = os.path.join(PREFIX_PATH, "model")
PRETRAINED_MODEL_PATH = os.path.join(PREFIX_PATH, "pretrained")
LABEL_PATH = os.path.join(MODEL_PATH, "label_map.pbtxt")

DEFAULT_MODEL = "faster_rcnn_resnet101_coco"

PRETRAINED_MODELS = [
    "ssd_mobilenet_v1_coco.pb",
    "ssd_mobilenet_v2_coco.pb",
    "faster_rcnn_resnet101_coco.pb",
    "rfcn_resnet101_coco.pb",
    "faster_rcnn_inception_v2_coco.pb",
    "ssd_inception_v2_coco.pb",
    "faster_rcnn_resnet50_coco.pb",
    "faster_rcnn_resnet50_lowproposals_coco.pb",
    "faster_rcnn_resnet101_lowproposals_coco.pb",
    "faster_rcnn_inception_resnet_v2_atrous_coco.pb",
    "faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco.pb",
    "faster_rcnn_nas_coco.pb",
    "faster_rcnn_nas_lowproposals_coco.pb",
]


class Configuration:
    def __init__(self, data: dict):
        self.file = data.get("file", None)
        self.image = data.get("image", None)
        self.stride = data.get("stride", 1)

        self.model = data.get("model", DEFAULT_MODEL)

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
    models = {}

    @staticmethod
    def factory(configuration: Configuration):
        if configuration.image is not None or (
            configuration.file is not None
            and configuration.file.lower().endswith((".jpg", ".png", ".jpeg"))
        ):
            return ImageProcessor(configuration, Cache(configuration))

        return None

    @staticmethod
    def get_model(frozen_graph=DEFAULT_MODEL):
        # If the frozen graph was specified without the extension, let's
        # set it here.
        if not frozen_graph.lower().endswith(".pb"):
            frozen_graph += ".pb"

        # Now we need to load the protobuf frozen graph. To do that we
        # need to determine whether the model is one of the pre-trained
        # models that come with the container or a new model specified
        # by the user.
        model_path = (
            os.path.join(PRETRAINED_MODEL_PATH, frozen_graph)
            if frozen_graph in PRETRAINED_MODELS
            else os.path.join(MODEL_PATH, frozen_graph)
        )

        if frozen_graph not in Processor.models:
            Processor.models[frozen_graph] = Model(LABEL_PATH, model_path)

        return Processor.models[frozen_graph]

    def __init__(self, configuration: Configuration, cache: Cache):
        self.configuration = configuration
        self.cache = cache

    def inference(self):
        # This method should be implemented on the sub-classes.
        pass

    def predict(self, image):
        return Processor.get_model(self.configuration.model).inference(image)


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
            inference = self.predict(image)

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
                    elif fragments.scheme == "http" or fragments.scheme == "https":
                        image = self.__get_image_from_url(self.__get_source())
                    else:
                        image = self.__get_image_from_file(
                            self.__get_source(), fragments
                        )

                self.cache.put(source_cache_key, image)
            except Exception as e:
                logging.error("There was an error handling image", e)
        else:
            logging.debug(f"Image found in cache")

        return image

    def __get_image_from_base64_string(self, source) -> np.array:
        logging.info(f"Creating image from base64 string...")

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

    def __get_image_from_url(self, file) -> np.array:
        logging.info(f"Downloading image from URL. Filename {file}...")

        tmp = tempfile.NamedTemporaryFile()

        with open(tmp.name, "wb") as f:
            response = requests.get(file, stream=True)

            if not response.ok:
                raise RuntimeError(
                    f"There was an error downloading image from URL. Filename {file}. Response {response}."
                )

            for block in response.iter_content(1024):
                if not block:
                    break
                f.write(block)

            return self.__numpy(Image.open(tmp.name))

    def __get_image_from_file(self, file, fragments) -> np.array:
        logging.info(f"Downloading image from file. Filename {file}...")

        if fragments.scheme == "file":
            return self.__numpy(Image.open(fragments.path))

        return self.__numpy(Image.open(file))

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


@app.route("/invocations", methods=["POST"])
def invoke():
    """
    TODO:

    * Add support to provide a threshold.
    * What happens if there are no detections?
    * Is the "return only these classes" working?
    * If I already have cache for the final result, why do we need to cache the file?

    * Add support to multiple (batch) images: https://stackoverflow.com/questions/49750520/run-inference-for-single-imageimage-graph-tensorflow-object-detection

    * Add support to provide a video: file | stride

    * Implement gRPC interface
    
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
