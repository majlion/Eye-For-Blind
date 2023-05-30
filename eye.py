import numpy as np
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import tensorflow as tf

# Path to the frozen detection graph and label map
PATH_TO_FROZEN_GRAPH = './frozen_inference_graph.pb'
PATH_TO_LABEL_MAP = './label_map.pbtxt'

# Number of classes in the label map
NUM_CLASSES = 90

# Load the frozen detection graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load the label map
label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Perform object detection
def detect_objects(image_path):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_np = cv2.imread(image_path)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            cv2.imshow('Object Detection', cv2.resize(image_np, (800, 600)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Run object detection on an image
detect_objects('path_to_image/image.jpg')
