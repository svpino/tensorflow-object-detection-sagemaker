import uuid
import string
import unicodedata


class Configuration:
    """
    Configuration object holding all the available settings to instantiate
    a :class:`~doodl.model.Model` object.
    """

    _VALID_CHARACTERS = "-_.()%s%s" % (string.ascii_letters, string.digits)

    def __init__(self, **kwargs):
        self.endpoint = kwargs.get("endpoint", None)
        self.model = kwargs.get("model", None)

        self.cache = kwargs.get("cache", False)
        self.cache_hash = kwargs.get("cache_hash", uuid.uuid4().hex)

        self.aws_region = kwargs.get("aws_region", None)
        self.aws_access_key = kwargs.get("aws_access_key", None)
        self.aws_secret_access_key = kwargs.get("aws_secret_access_key", None)

    @property
    def endpoint(self):
        """
        The endpoint to connect to the object detection backend server,
        if any. This is `None` by default indicating that a local object detection
        implementation should be used.
        """
        return self.__endpoint

    @endpoint.setter
    def endpoint(self, value):
        self.__endpoint = value

    @property
    def model(self):
        """
        The name of the pre-trained model file that should be loaded and used to
        run the object detection process, or a URL pointing to the frozen inference
        graph file in `.tar.gz` format.

        Doodl supports a list of pre-trained models that can be specified directly
        using this attribute:

        * ssd_mobilenet_v1_coco_2018_01_28
        * ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03
        * ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18
        * ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18
        * ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03
        * ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
        * ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
        * ssd_mobilenet_v2_coco_2018_03_29
        * ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03
        * ssd_inception_v2_coco_2018_01_28
        * faster_rcnn_inception_v2_coco_2018_01_28
        * faster_rcnn_resnet50_coco_2018_01_28
        * faster_rcnn_resnet50_lowproposals_coco_2018_01_28
        * rfcn_resnet101_coco_2018_01_28
        * faster_rcnn_resnet101_coco_2018_01_28
        * faster_rcnn_resnet101_lowproposals_coco_2018_01_28
        * faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
        * faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28
        * faster_rcnn_nas_coco_2018_01_28
        * faster_rcnn_nas_lowproposals_coco_2018_01_28
        * faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28
        * faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28
        * facessd_mobilenet_v2_quantized_320x320_open_image_v4
        * faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12
        * ssd_mobilenet_v2_oid_v4_2018_12_12
        * ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20
        """
        return self.__model

    @model.setter
    def model(self, value):
        if not value:
            value = "faster_rcnn_inception_v2_coco_2018_01_28"

        self.__model = value

    @property
    def cache(self):
        """
        Whether the object detection process should cache detections. By default
        this attribute is **False**.

        Cached predictions are organized in folders determined by the
        :attr:`~ cache_cash` property. Entries are either saved inside the
        ``~/.doodl/cache`` or ``/tmp/.doodl/cache`` directory.
        """
        return self.__cache

    @cache.setter
    def cache(self, value):
        if not value:
            value = False

        self.__cache = value

    @property
    def cache_hash(self):
        """
        Represent the hash key used to store predictions in the cache. Cached entries
        will be organized using this hash value. This is useful when trying to
        bypass previously cached results, or when manually removing entries from the
        cache. By default, if no hash is specified, a new ``uuid.uuid4().hex`` is
        assigned to this property.
        """
        return self.__cache_hash

    @cache_hash.setter
    def cache_hash(self, value):
        if not value:
            value = uuid.uuid4().hex

        value = unicodedata.normalize("NFKD", value).encode("ASCII", "ignore")
        value = "".join(
            chr(c) for c in value if chr(c) in Configuration._VALID_CHARACTERS
        )

        self.__cache_hash = value
