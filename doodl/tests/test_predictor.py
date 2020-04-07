import pytest
import numpy as np

from doodl import Predictor, Configuration, ImagePredictor, Cache

from mocks import MockBackend1, MockBackend2


class MockCache(Cache):
    def __init__(self, configuration: Configuration, value=None):
        super(MockCache, self).__init__(configuration)
        self.value = value

    def get(self, key: str):
        return self.value


class MockImagePredictor1(ImagePredictor):
    def __init__(self, configuration: Configuration, cache: Cache, image=None):
        super(MockImagePredictor1, self).__init__(configuration, cache)
        self.image = image

    def _predict(self, image):
        return {
            "detection_classes": [1, 2, 3],
            "detection_boxes": np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]),
            "detection_scores": np.array([0.9, 0.8, 0.7]),
        }

    def _get_image(self, source):
        return self.image


class MockImagePredictor2(ImagePredictor):
    def _get_image_from_s3(self, source):
        return "s3"

    def _get_image_from_url(self, source):
        return "url"

    def _get_image_from_file(self, source):
        return "file"

    def _get_image_from_base64(self, source):
        return "base64"


def test_predictor_invalid_backend_raises_error():
    configuration = Configuration({"backend": "invalid-backend"})

    with pytest.raises(RuntimeError) as excinfo:
        Predictor(configuration, cache=Cache(configuration))

    assert "invalid-backend" in str(excinfo.value)


def test_predictor_supports_custom_cache():
    configuration = Configuration({"backend": "test_predictor.MockBackend1"})
    predictor = ImagePredictor(configuration, MockCache(configuration))

    assert isinstance(predictor.cache, MockCache)


def test_image_returns_cached_predictions():
    configuration = Configuration(
        {"backend": "test_predictor.MockBackend1", "cache": True}
    )

    predictor = ImagePredictor(
        configuration, MockCache(configuration, value=np.array("123"))
    )
    assert predictor.inference(source="file.jpg") == "123"


def test_image_runs_prediction_if_not_cached():
    configuration = Configuration(
        {"backend": "test_predictor.MockBackend1", "cache": True}
    )

    predictor = MockImagePredictor1(
        configuration, MockCache(configuration), image=np.array([0, 1, 2]),
    )

    predictions = predictor.inference(source="file.jpg")

    assert "predictions" in predictions
    assert len(predictions["predictions"]) == 3

    assert predictions["predictions"][0][0] == 0.0
    assert predictions["predictions"][1][0] == 1.0
    assert predictions["predictions"][2][0] == 2.0

    assert predictions["predictions"][0][1] == 0.9
    assert predictions["predictions"][1][1] == 0.8
    assert predictions["predictions"][2][1] == 0.7

    assert predictions["predictions"][0][2] == 2.0
    assert predictions["predictions"][1][2] == 3.0
    assert predictions["predictions"][2][2] == 4.0

    assert predictions["predictions"][0][3] == 1.0
    assert predictions["predictions"][1][3] == 2.0
    assert predictions["predictions"][2][3] == 3.0

    assert predictions["predictions"][0][4] == 4.0
    assert predictions["predictions"][1][4] == 5.0
    assert predictions["predictions"][2][4] == 6.0

    assert predictions["predictions"][0][5] == 3.0
    assert predictions["predictions"][1][5] == 4.0
    assert predictions["predictions"][2][5] == 5.0


def test_get_image_supports_s3():
    configuration = Configuration(
        {"backend": "test_predictor.MockBackend1", "cache": True}
    )

    predictor = MockImagePredictor2(configuration, MockCache(configuration))

    assert predictor._get_image("s3://bucket/file.jpg") == 's3'


def test_get_image_supports_url():
    configuration = Configuration(
        {"backend": "test_predictor.MockBackend1", "cache": True}
    )

    predictor = MockImagePredictor2(configuration, MockCache(configuration))

    assert predictor._get_image("http://domain.com/file.jpg") == 'url'
    assert predictor._get_image("https://domain.com/file.jpg") == 'url'


def test_get_image_supports_file():
    configuration = Configuration(
        {"backend": "test_predictor.MockBackend1", "cache": True}
    )

    predictor = MockImagePredictor2(configuration, MockCache(configuration))

    assert predictor._get_image("file:///etc/tmp/file.jpg") == 'file'


def test_get_image_supports_base64():
    configuration = Configuration(
        {"backend": "test_predictor.MockBackend1", "cache": True}
    )

    predictor = MockImagePredictor2(configuration, MockCache(configuration))

    assert predictor._get_image("base64-string-here") == 'base64'


def test_get_image_supports_ndarray():
    configuration = Configuration(
        {"backend": "test_predictor.MockBackend1", "cache": True}
    )

    predictor = ImagePredictor(configuration, MockCache(configuration))

    result = predictor._get_image(np.array([[0, 1, 2], [1, 2, 3]]))

    assert isinstance(result, np.ndarray)
