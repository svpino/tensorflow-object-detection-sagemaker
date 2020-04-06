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

from todl import Processor, ImageProcessor, Configuration


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
    * Should cache key should include model that provided results?
    * Handle exceptions. They should bubble to the REST api response
    * What happens if there are no detections?
    * Is the "return only these classes" working?
    * If I already have cache for the final result, why do we need to cache the file?

    * Add support to provide a threshold.

    ##### Implement interface library to access backend #####

    * Interface should allow to use classes directly (when installed on same notebook)
    * Interface should allow to connect to container hosted somewhere

    configuration = Configuration(
        protocol=Protocol.gRPC,
        endpoint="127.0.0.1:67878",
        model="rfcn_resnet101_coco.pb",
        aws_region="us-east-1",
        aws_access_key="key",
        aws_secret_access_key="secret",
        cache=True
    )

    detector = Detector(configuration)

    detections = detector.inference(file=["s3://vinsa-object-detection/test2.jpg"])

    ###################################

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
