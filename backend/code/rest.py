import os
import io
import json
import base64
import logging
import logging.config
import numpy as np

from urllib.parse import urlparse
from flask import Flask, request, Response, jsonify
from PIL import Image
from PIL import ImageFile

from doodl import Model, Configuration, NumpyJsonSerializer

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# gunicorn_error_logger = logging.getLogger('gunicorn.error')
# app.logger.handlers.extend(gunicorn_error_logger.handlers)
# app.logger.setLevel(logging.DEBUG)

PREFIX_PATH = "/opt/ml/"
CACHE_PATH = os.path.join(PREFIX_PATH, "cache")
MODEL_PATH = os.path.join(PREFIX_PATH, "model")


@app.route("/ping", methods=["GET"])
def ping():
    """This endpoint determines whether the container is working and healthy.
    """
    logging.info("Ping received...")

    # health = Processor.get_model() is not None
    health = True

    status = 200 if health else 404
    return Response(response="\n", status=status, mimetype="application/json")


@app.route("/inference", methods=["POST"])
@app.route("/invocations", methods=["POST"])
def invoke():
    """
    TODO:
    * What happens if there are no detections?
    * Test container
    * By default, MODEL_SERVER_WORKERS should be 1 if not specified

    * Check that we are exporting the right modules:
        import re
        print(dir(re))

    * Should we print the results on the backend logs?

    * Add GPU support

    * Clean README.md files
    * Link to repo from documentation
    * fix reference: (version 0.1 specified on installation)
    * I should link to the Docker Hub page from docs (installation)
    * Prepare notebook with example on how to use this.

    --- v2 ---

    * Add support to multiple (batch) images: https://stackoverflow.com/questions/49750520/run-inference-for-single-imageimage-graph-tensorflow-object-detection
    * Add support to provide a video: file | stride
    * Implement gRPC interface
    * Add Docker Hub description
    * add visualization function to the library (draw boxes on image)  
    * Add support to provide a threshold.

    --- v3 ---

    * Optimize Docker image (it's sitting right now at 5.09 GB)

    """

    if request.content_type == "application/json":
        data = request.get_json()

        configuration = Configuration(**data)
        configuration.endpoint = None

        logging.debug(f"Configuration: {configuration.__dict__}")

        source = data.get("source", None)
        if source is None:
            return (
                jsonify(
                    reason=('Request does not contain attribute "source".'),
                    mimetype="application/json",
                ),
                400,
            )

        source = __get_source(source)

        try:
            model = Model(configuration)
            result = model.inference(source)
        except Exception as e:
            return (
                jsonify(
                    reason=(
                        "There was an error running inference on the supplied "
                        "source data."
                    ),
                    exception=str(e),
                    mimetype="application/json",
                ),
                500,
            )

        logging.debug(f"Inference result {result}")

        return Response(
            response=json.dumps(result), status=200, mimetype="application/json",
        )

    return (
        jsonify(
            reason=("Request is not application/json"), mimetype="application/json",
        ),
        400,
    )


def __get_source(source):
    if source and "__ndarray__" in source:
        logging.debug("Source data is numpy.ndarray")
        return json.loads(source, object_hook=NumpyJsonSerializer.decoder)

    fragments = urlparse(source, allow_fragments=False)
    if fragments.scheme in ("http", "https", "s3"):
        logging.debug("Source data is URL")
        return source

    try:
        image_data = base64.b64decode(source)
        image = Image.open(io.BytesIO(image_data))
        (width, height) = image.size
        return np.array(image.getdata()).reshape((height, width, 3)).astype(np.uint8)
    except Exception:
        raise RuntimeError("There was an error decoding the source object.")
