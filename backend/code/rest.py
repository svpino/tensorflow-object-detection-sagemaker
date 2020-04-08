import os
import json
import logging
import logging.config

from flask import Flask, request, Response, jsonify

from doodl import Model, Configuration, NumpyJsonSerializer


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# gunicorn_error_logger = logging.getLogger('gunicorn.error')
# app.logger.handlers.extend(gunicorn_error_logger.handlers)
# app.logger.setLevel(logging.DEBUG)

PREFIX_PATH = "/opt/ml/"
CACHE_PATH = os.path.join(PREFIX_PATH, "cache")
MODEL_PATH = os.path.join(PREFIX_PATH, "model")
PRETRAINED_MODEL_PATH = os.path.join(PREFIX_PATH, "pretrained")

DEFAULT_MODEL = "faster_rcnn_inception_v2_coco.pb"

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
    * Should cache key should include model that provided results?
    * How should we use the cache_id? Maybe as a folder?
    * What happens if there are no detections?
    * Is the "return only these classes" working?

    * Add support to provide a threshold.

    * Check that we are exporting the right modules:
        import re
        print(dir(re))

    * Should we print the results on the backend logs?

    * Prepare notebook with example on how to use this.

    --- v2 ---

    * Add support to multiple (batch) images: https://stackoverflow.com/questions/49750520/run-inference-for-single-imageimage-graph-tensorflow-object-detection
    * Add support to provide a video: file | stride
    * Implement gRPC interface

    """

    if request.content_type == "application/json":
        data = request.get_json()

        configuration = Configuration(data)
        __update_model_reference(configuration)
        configuration.endpoint = None
        configuration.cache_path = CACHE_PATH

        logging.debug(f"Configuration: {configuration.__dict__}")

        source = __get_source(data.get("source", None))
        if source is None:
            return (
                jsonify(
                    reason=('Request does not contain attribute "source".'),
                    mimetype="application/json",
                ),
                400,
            )

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

    return source


def __update_model_reference(configuration):
    if not configuration.model:
        configuration.model = DEFAULT_MODEL

    if not configuration.model.lower().endswith(".pb"):
        configuration.model += ".pb"

    if configuration.model in PRETRAINED_MODELS:
        configuration.model_path = PRETRAINED_MODEL_PATH
    elif not configuration.model_path:
        configuration.model_path = MODEL_PATH
