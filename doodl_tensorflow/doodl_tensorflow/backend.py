import os
import logging
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util


class TensorflowBackend:
    def __init__(self, model, model_path, model_label_path, **kwargs):
        self.model = model
        self.model_path = os.path.join(model_path, self.model)
        self.label_path = model_label_path

        self.__load_graph()

    def inference(self, image):
        ops = self.global_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            "num_detections",
            "detection_boxes",
            "detection_scores",
            "detection_classes",
        ]:
            tensor_name = key + ":0"
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.global_graph.get_tensor_by_name(tensor_name)

        image_tensor = self.global_graph.get_tensor_by_name("image_tensor:0")

        logging.info("Running inference on image...")
        detections = self.session.run(
            tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)}
        )

        num_detections = int(detections["num_detections"][0])
        detections["num_detections"] = num_detections
        detections["detection_classes"] = detections["detection_classes"][0][
            0:num_detections
        ].astype(np.uint8)
        detections["detection_boxes"] = detections["detection_boxes"][0][
            0:num_detections
        ]
        detections["detection_scores"] = detections["detection_scores"][0][
            0:num_detections
        ]

        return detections

    def __load_graph(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default() as default_graph:
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True

        self.category_index = label_map_util.create_category_index_from_labelmap(
            self.label_path, use_display_name=True
        )
        self.session = tf.Session(config=config, graph=default_graph)
        self.global_graph = default_graph
