import os
from doodl import Cache, Configuration


def test_cache_is_disbled_by_default():
    cache = Cache(Configuration())
    assert cache.enabled is False


def test_put_get():
    cache = Cache(Configuration({"cache": True}))

    cache.put("key", "value")

    assert os.path.exists("key.npz") is True

    value = cache.get("key")
    assert value == "value"

    os.remove("key.npz")


def test_put_shouldnt_cache_if_disabled():
    cache = Cache(Configuration({"cache": False}))

    cache.put("key", "value")
    assert os.path.exists("key.npz") is False


def test_get_should_return_none_if_disabled():
    cache = Cache(Configuration({"cache": True}))

    cache.put("key", "value")
    assert os.path.exists("key.npz") is True

    cache.enabled = False
    assert cache.get("key") is None

    os.remove("key.npz")


def test_cache_path_is_set_to_local_folder_by_default():
    cache = Cache(Configuration({"cache": True}))
    assert cache.cache_path == "."
