import os
import json
import requests
import base64
import numpy as np

from urllib.parse import urlparse

from doodl import Configuration, ImagePredictor, Cache, NumpyJsonSerializer


class Model:
    """
    Represents an Object Detection model capable of running inference on images.
    This class is the main interaction point with the object detection backend.

    Attributes:
        configuration(doodl.configuration.Configuration): The configuration settings
            used to initialize this model.

    Args:
        configuration(doodl.configuration.Configuration, optional): The configuration
            that should be used to set up the model, defaults to None
    """

    def __init__(self, configuration: Configuration = None):
        self.configuration = configuration or Configuration()

    def inference(self, source):
        """
        Runs the object detection process on the provided source and returns the list
        of detections.

        The format of the output is a dictionary with a single *predictions* entry
        containing a 2-dimensional array where every row represents a single detection,
        and six columns with following values:

        * Column 0: The identifier of the detected class.
        * Column 1: The confidence score on the detection.
        * Column 2: The *xmin* normalized value of the detected box.
        * Column 3: The *ymin* normalized value of the detected box.
        * Column 4: The *xmax* normalized value of the detected box.
        * Column 5: The *ymax* normalized value of the detected box.

        Example:
            Here is an example of a result including two detections::

                {
                    'predictions': [
                        [0, 0.99, 0.18, 0.18, 0.36, 0.68],
                        [2, 0.99, 0.52, 0.30, 0.92, 0.63]
                    ]
                }

        Args:
            source(object): The image where we want to run inference. This argument
                supports an HTTP/S or AWS S3 URL pointing to the object, the path of
                the file, or a 3-dimensional numpy array representing the image.
        Returns:
            dict: A dictionary with a single *predictions* entry containing the list
                of detected objects found in the source image.
        Raises:
            RuntimeError: if an error happens while running the object detection
                process.
        """

        if source is None:
            raise ValueError('The "source" attribute shouldn\'t be empty')

        if self.configuration.endpoint and self.configuration.endpoint.startswith(
            ("http://", "https://")
        ):

            response = requests.post(
                self.configuration.endpoint,
                json=dict(
                    **{"source": self.__get_serialized_source(source)},
                    **self.configuration.__dict__
                ),
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code in (400, 500):
                raise RuntimeError(response.json()["reason"])
            else:
                raise RuntimeError(response.text)
        else:
            predictor = ImagePredictor(self.configuration, Cache(self.configuration))
            return predictor.inference(source)

    def __get_serialized_source(self, source):
        # If the source is specified as a numpy array, we need to serialize it
        # to JSON.
        if isinstance(source, np.ndarray):
            return json.dumps(source, cls=NumpyJsonSerializer)

        if isinstance(source, str):
            fragments = urlparse(source, allow_fragments=False)
            if fragments.scheme in ("http", "https", "s3"):
                return source

            # At this point we can assume the image is a local file
            if os.path.exists(source):
                with open(source, "rb") as image:
                    encoded_string = base64.b64encode(image.read())

                return encoded_string.decode("utf-8")

            raise RuntimeError(
                "There was an error interpreting the source object. "
                "Make sure you are specifying a valid URL, path, or numpy array."
            )
