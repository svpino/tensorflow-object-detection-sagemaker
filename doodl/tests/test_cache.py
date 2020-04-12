import os
from doodl import Cache, Configuration


def test_cache_is_disbled_by_default():
    cache = Cache(Configuration())
    assert cache.enabled is False


def test_put_get():
    configuration = Configuration(cache=True)
    cache = Cache(configuration)

    cache.put("key", "value")
    cache_file_path = os.path.join(
        cache.cache_path, configuration.cache_hash, "key.npz"
    )
    assert os.path.exists(cache_file_path) is True

    value = cache.get("key")
    assert value == "value"

    os.remove(cache_file_path)


def test_put_shouldnt_cache_if_disabled():
    configuration = Configuration(cache=False)
    cache = Cache(configuration)

    cache.put("key", "value")
    cache_file_path = os.path.join(
        cache.cache_path, configuration.cache_hash, "key.npz"
    )
    assert os.path.exists(cache_file_path) is False


def test_get_should_return_none_if_disabled():
    configuration = Configuration(cache=True)
    cache = Cache(configuration)

    cache.put("key", "value")
    # assert os.path.exists(cache_file_path) is True
    assert cache.get("key") == "value"

    cache.enabled = False
    assert cache.get("key") is None

    cache_file_path = os.path.join(
        cache.cache_path, configuration.cache_hash, "key.npz"
    )
    os.remove(cache_file_path)
