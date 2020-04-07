import json
import ast
import base64
import requests
import numpy as np

from doodl import Configuration, ImagePredictor, Cache


class NumpyJsonSerializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            data = base64.b64encode(np.ascontiguousarray(obj).data)
            return dict(
                __ndarray__=data.decode("utf-8"), dtype=str(obj.dtype), shape=obj.shape
            )

        return json.JSONEncoder.default(self, obj)

    @staticmethod
    def decoder(obj):
        source = obj
        if isinstance(source, str):
            try:
                source = ast.literal_eval(obj)
            except Exception:
                return obj

        if isinstance(source, dict) and "__ndarray__" in source:
            data = base64.b64decode(source["__ndarray__"])
            return np.frombuffer(data, source["dtype"]).reshape(source["shape"])

        return source


class Model:
    """
    This is the Model class.


    # TODO:

    * Add new endpoint (inference) to REST backend.

    """

    def __init__(self, configuration: Configuration):
        self.configuration = configuration or Configuration()

    def inference(self, source):
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
        # If the source is specified as a Numpy array, we need to serialize it
        # to JSON.
        if isinstance(source, np.ndarray):
            return json.dumps(source, cls=NumpyJsonSerializer)

        return source
