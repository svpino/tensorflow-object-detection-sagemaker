class Configuration:
    """
    Configuration object holding all the available settings to instantiate
    a :class:`~doodl.model.Model` object.

    Attributes:
        endpoint : :obj:`str`, optional
            The endpoint to connect to the object detection backend server,
            if any. This is `None` by default indicating that a local object detection
            implementation should be used.
        backend : :obj:`str`, optional
            The backend implementation of the object detection process. By
            default this attribute has the value `tensorflow` which is currently
            the only supported backend.
        model : :obj:`str`, optional
            The name of the model file that should be loaded and used to run the object
            detection process. When used with a backend server implementation, this
            attribute defaults to `faster_rcnn_inception_v2_coco`. When used
            with a local process, this value defaults to `frozen_inference_graph.pb`.

            The backend server comes out of the box with several pretrained models. 
            To use one of these models, you can set this attribute to one of the
            following values:

            * ssd_mobilenet_v1_coco
            * ssd_mobilenet_v2_coco
            * faster_rcnn_resnet101_coco
            * rfcn_resnet101_coco
            * faster_rcnn_inception_v2_coco
            * ssd_inception_v2_coco
            * faster_rcnn_resnet50_coco
            * faster_rcnn_resnet50_lowproposals_coco
            * faster_rcnn_resnet101_lowproposals_coco
            * faster_rcnn_inception_resnet_v2_atrous_coco
            * faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco
            * faster_rcnn_nas_coco
            * faster_rcnn_nas_lowproposals_coco

            None of these pretained models are available when using a local
            implementation. In this case you are responsible for setting up the model
            using :attr:`model` in combination with :attr:`model_path`.
        model_path : :obj:`str`, optional
            The path where the :attr:`model` file can be found. When using a backend
            server and :attr:`model` is one of the included pretrained models, this
            value will be ignored because the pretrained model will be automatically
            loaded. If :attr:`model` is not one of the pretrained models, and this
            value is not specified, this path will default to `/opt/ml/model`.
        cache : :obj:`bool`, optional
            Whether the object detection process should cache detections. By default
            this attribute is **False**.
        cache_path : :obj:`str`, optional
            The path where cache objects will be stored. The default value of this
            attribute is `/opt/ml/cache`.
    """

    def __init__(self, data: dict = None):
        self.endpoint = data.get("endpoint", None) if data else None
        self.backend = data.get("backend", None) if data else None

        self.model = data.get("model", None) if data else None
        self.model_path = data.get("model_path", None) if data else None

        self.cache = data.get("cache", None) if data else None
        self.cache_path = data.get("cache_path", None) if data else None
        self.cache_id = data.get("cache_id", None) if data else None

        self.aws_region = data.get("aws_region", None) if data else None
        self.aws_access_key = data.get("aws_access_key", None) if data else None
        self.aws_secret_access_key = (
            data.get("aws_secret_access_key", None) if data else None
        )
