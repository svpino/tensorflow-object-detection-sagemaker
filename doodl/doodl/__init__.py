import json
import ast
import base64
import numpy as np

from .configuration import Configuration
from .cache import Cache
from .backend import Backend
from .predictor import Predictor, ImagePredictor


class NumpyJsonSerializer(json.JSONEncoder):
    """
    Serializer/deserializer to properly encode numpy arrays back and forth
    to and from JSON objects.

    This class is used internally by :class:`~doodl.model.Model`, and its not
    intended (or probably necessary) to be used by users of the library.

    Example:
        An example of using this class is as follows::

            encoded_json = json.dumps(array, cls=NumpyJsonSerializer)

        This assumes that `array` is a numpy array.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            data = base64.b64encode(np.ascontiguousarray(obj).data)
            return dict(
                __ndarray__=data.decode("utf-8"), dtype=str(obj.dtype), shape=obj.shape
            )

        return json.JSONEncoder.default(self, obj)

    @staticmethod
    def decoder(obj):
        """
        Object hook implementation to convert an encoded numpy array back into
        its original format.

        Example:
            You can use this function the following way::

                array = json.loads(
                    encoded_json,
                    object_hook=NumpyJsonSerializer.decoder)

            This assumes that the `encoded_json` contains a JSON object that was
            previously encoded using this class.
        """

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


from .model import Model, NumpyJsonSerializer


