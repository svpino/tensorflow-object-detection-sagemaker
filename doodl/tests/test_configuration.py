from doodl import Configuration


def test_model_default_value():
    configuration = Configuration()
    assert configuration.model == "faster_rcnn_inception_v2_coco_2018_01_28"


def test_model_default_value_if_blank():
    configuration = Configuration()
    configuration.model = ""
    assert configuration.model == "faster_rcnn_inception_v2_coco_2018_01_28"


def test_cache_is_false_by_default():
    configuration = Configuration()
    assert configuration.cache is False

    configuration = Configuration(cache=True)
    assert configuration.cache is True


def test_cache_hash_defaults_to_uuid():
    configuration = Configuration()
    assert len(configuration.cache_hash) == 32


def test_cache_hash_set_to_uuid_if_empty():
    configuration = Configuration(cache_hash="")
    assert len(configuration.cache_hash) == 32


def test_cache_hash_should_sanitize_input_value():
    configuration = Configuration(cache_hash="this is * a = crazy \\ key")
    assert configuration.cache_hash == "thisisacrazykey"
