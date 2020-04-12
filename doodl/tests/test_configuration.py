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
