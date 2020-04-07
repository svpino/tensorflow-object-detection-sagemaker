import io
import hashlib
import logging
import logging.config
import boto3
import tempfile
import requests
import numpy as np

from urllib.parse import urlparse

from PIL import Image
from PIL import ImageFile

from doodl.backend import Backend
from doodl import Configuration, Cache


ImageFile.LOAD_TRUNCATED_IMAGES = True


class Predictor:
    def __init__(self, configuration: Configuration, cache: Cache):
        self.backend = Backend.register(configuration.backend, configuration)
        self.configuration = configuration
        self.cache = cache

    def inference(self, source):
        pass

    def _predict(self, image):
        return self.backend.inference(image)


class ImagePredictor(Predictor):
    def __init__(self, configuration: Configuration, cache: Cache):
        super(ImagePredictor, self).__init__(configuration, cache)

    def inference(self, source):
        logging.info(f"Running inference on image...")

        inference_cache_key = self.__get_key(source, "inference")

        predictions = None

        value = self.cache.get(inference_cache_key)
        if value is not None:
            predictions = value.tolist()
            logging.info(f"Inference from image found in cache")

        if predictions is None:
            logging.info(f"Inference from image not found in cache")

            image = self._get_image(source)
            inference = self._predict(image)

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

    def _get_image(self, source):
        if isinstance(source, np.ndarray):
            return source

        image = None
        fragments = urlparse(source, allow_fragments=False)

        if fragments.scheme == "s3":
            image = self._get_image_from_s3(source)
        elif fragments.scheme == "http" or fragments.scheme == "https":
            image = self._get_image_from_url(source)
        elif fragments.scheme == "file":
            image = self._get_image_from_file(source)
        else:
            image = self._get_image_from_base64(source)

        return image

    def __get_key(self, source: str, suffix: str):
        tmp = hashlib.md5(str(source).encode("utf8"))
        tmp.update(str(suffix).encode("utf8"))
        return tmp.hexdigest()

    def _get_image_from_base64(self, source) -> np.array:
        logging.info(f"Creating image from base64 string...")

        image = Image.open(io.BytesIO(source))
        return self.__numpy(image)

    def _get_image_from_s3(self, source) -> np.array:
        logging.info(f"Downloading image from S3. Filename {source}...")
        fragments = urlparse(source, allow_fragments=False)

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

    def _get_image_from_url(self, source) -> np.array:
        logging.info(f"Downloading image from URL. Filename {source}...")

        tmp = tempfile.NamedTemporaryFile()

        with open(tmp.name, "wb") as f:
            response = requests.get(source, stream=True)

            if not response.ok:
                raise RuntimeError(
                    f"There was an error downloading image from URL. Filename {source}. "
                    f"Response {response}."
                )

            for block in response.iter_content(1024):
                if not block:
                    break
                f.write(block)

            return self.__numpy(Image.open(tmp.name))

    def _get_image_from_file(self, source) -> np.array:
        logging.info(f"Downloading image from file. Filename {source}...")
        fragments = urlparse(source, allow_fragments=False)

        if fragments.scheme == "file":
            return self.__numpy(Image.open(fragments.path))

        return self.__numpy(Image.open(source))

    def __numpy(self, image):
        (width, height) = image.size
        return np.array(image.getdata()).reshape((height, width, 3)).astype(np.uint8)
