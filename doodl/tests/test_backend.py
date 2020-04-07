from doodl import Backend, Configuration, Predictor, Cache

from mocks import MockBackend1, MockBackend2


def test_register_backend():
    configuration1 = Configuration({"backend": "test_backend.MockBackend1"})

    configuration2 = Configuration({"backend": "test_backend.MockBackend2"})

    Predictor(configuration1, cache=Cache(configuration1))
    Predictor(configuration2, cache=Cache(configuration2))

    assert "test_backend.MockBackend1" in Backend.backends
    assert "test_backend.MockBackend2" in Backend.backends
    assert isinstance(Backend.backends["test_backend.MockBackend1"], MockBackend1)
    assert isinstance(Backend.backends["test_backend.MockBackend2"], MockBackend2)
