import os
import tarfile
import shutil

from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve

from doodl import Configuration


class Backend:
    backends = {}

    PRETRAINED_MODELS = [
        "ssd_mobilenet_v1_coco_2018_01_28",
        "ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03",
        "ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18",
        "ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18",
        "ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03",
        "ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03",
        "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03",
        "ssd_mobilenet_v2_coco_2018_03_29",
        "ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03",
        "ssd_inception_v2_coco_2018_01_28",
        "faster_rcnn_inception_v2_coco_2018_01_28",
        "faster_rcnn_resnet50_coco_2018_01_28",
        "faster_rcnn_resnet50_lowproposals_coco_2018_01_28",
        "rfcn_resnet101_coco_2018_01_28",
        "faster_rcnn_resnet101_coco_2018_01_28",
        "faster_rcnn_resnet101_lowproposals_coco_2018_01_28",
        "faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28",
        "faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28",
        "faster_rcnn_nas_coco_2018_01_28",
        "faster_rcnn_nas_lowproposals_coco_2018_01_28",
        "faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28",
        "faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28",
        "facessd_mobilenet_v2_quantized_320x320_open_image_v4",
        "faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12",
        "ssd_mobilenet_v2_oid_v4_2018_12_12",
        "ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20",
    ]

    @staticmethod
    def register(configuration: Configuration):
        if configuration.model in Backend.PRETRAINED_MODELS:
            model = configuration.model
            origin = "http://download.tensorflow.org/models/object_detection/"
        elif configuration.model.startswith(
            ("http://", "https://")
        ) and configuration.model.endswith(".tar.gz"):
            model = configuration.model[configuration.model.rfind("/") + 1 : -7]
            origin = configuration.model[: configuration.model.rfind("/") + 1]
        else:
            raise RuntimeError(
                f'The specified model "{configuration.model}" should be a valid'
                f"URL pointing to a .tar.gz file. "
            )

        model_filename = Backend._download_file(model=model, origin=origin)

        backend_key = f"tensorflow-{model_filename}"
        if backend_key not in Backend.backends:
            try:
                from doodl_tensorflow.backend import TensorflowBackend

                backend = TensorflowBackend(model=model_filename)
                Backend.backends[backend_key] = backend

            except Exception:
                raise RuntimeError(
                    "Unable to find tensorflow backend implementation. "
                    "Make sure you have 'doodl[tensorflow]' installed"
                )

        return Backend.backends[backend_key], model_filename

    @staticmethod
    def _download_file(model, origin, model_subdir="model", doodl_dir=None):
        """
        Downloads the specified pre-trained model file and copies it locally
        into doodl's base directory.

        The implementation of this method closely follows the code of the
        tf.keras.utils.get_file method.
        """

        if doodl_dir is None:
            if "DOODL_HOME" in os.environ:
                doodl_dir = os.environ.get("DOODL_HOME")
            else:
                doodl_dir = os.path.join(os.path.expanduser("~"), ".doodl")

        model_base_dir = os.path.expanduser(doodl_dir)

        if not os.access(model_base_dir, os.W_OK):
            model_base_dir = os.path.join("/tmp", ".doodl")

        model_dir = os.path.join(model_base_dir, model_subdir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_file_path = os.path.join(model_dir, f"{model}.pb")
        model_tar_path = os.path.join(model_dir, f"{model}.tar.gz")
        model_untarred_path = os.path.join(model_dir, model)

        error_message = "Failed fetching model file from {}. Error: {}. Message: {}"
        if not os.path.exists(model_file_path):
            try:
                try:
                    urlretrieve(
                        f"{origin}{model}.tar.gz",
                        os.path.join(model_dir, model_tar_path),
                    )
                except HTTPError as e:
                    raise RuntimeError(error_message.format(origin, e.code, e.msg))
                except URLError as e:
                    raise RuntimeError(error_message.format(origin, e.errno, e.reason))
            except (Exception, KeyboardInterrupt):
                if os.path.exists(model_tar_path):
                    os.remove(model_tar_path)
                raise

            if tarfile.is_tarfile(model_tar_path):
                with tarfile.open(model_tar_path) as archive:
                    try:
                        for member in archive.getmembers():
                            if "frozen_inference_graph.pb" in member.name:
                                archive.extract(member, model_dir)
                                break
                    except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                        if os.path.exists(model_tar_path):
                            os.remove(model_tar_path)

                        if os.path.exists(model_untarred_path):
                            shutil.rmtree(model_untarred_path)

                        raise RuntimeError(
                            f"Error extracting model file {model_tar_path}"
                        )

                    shutil.move(
                        os.path.join(model_untarred_path, "frozen_inference_graph.pb"),
                        model_file_path,
                    )

                    if os.path.exists(model_untarred_path):
                        shutil.rmtree(model_untarred_path)

                    if os.path.exists(model_tar_path):
                        os.remove(model_tar_path)

        return model_file_path

    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def inference(self, image):
        pass
