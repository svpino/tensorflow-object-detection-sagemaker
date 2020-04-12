import os
import pytest

from doodl import Backend, Configuration


def test_pretrained_model_is_downloaded():
    configuration = Configuration()

    backend, model_filename = Backend.register(configuration)
    assert os.path.exists(model_filename) is True

    from doodl_tensorflow.backend import TensorflowBackend
    assert isinstance(backend, TensorflowBackend) is True


def test_error_is_raised_if_model_is_not_targz():
    configuration = Configuration(model="https://www.example.com/file.zip")

    with pytest.raises(RuntimeError):
        Backend.register(configuration)


def test_error_is_raised_if_model_is_not_valid_url():
    configuration = Configuration(model="something-invalid")

    with pytest.raises(RuntimeError):
        Backend.register(configuration)


def test_error_is_raised_if_model_cant_be_downloaded():
    configuration = Configuration(model="https://www.example.com/file.tar.gz")

    with pytest.raises(RuntimeError):
        Backend.register(configuration)
