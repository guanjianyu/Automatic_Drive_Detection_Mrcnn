import numpy as np
import os
import tarfile
import cv2
import tensorflow as tf
from six.moves import urllib
from utils import label_map_util
from utils import visualization_utils as vis_util
from utils import ops as utils_ops

PROTOS_DIR = "C:/TensorFlow/models/research/object_detection/protos"
MIN_NUM_PY_FILES_IN_PROTOS_DIR = 5

# path to download respository
DOWNLOAD_MODEL_FROM_LOC = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
MODEL_FILE_NAME = MODEL_NAME + '.tar.gz'
MODEL_SAVE_DIR_LOC = "C:/TensorFlow/models/research/object_detection"

FROZEN_INFERENCE_GRAPH_LOC = MODEL_SAVE_DIR_LOC + "/" + MODEL_NAME + "/" + "frozen_inference_graph.pb"
# List of the strings that is used to add correct label for each box.
LABEL_MAP_LOC = "C:/TensorFlow/models/research/object_detection/data/mscoco_label_map.pbtxt"

NUM_CLASSES = 90


def download_model():
    try:
        if not os.path.exists(FROZEN_INFERENCE_GRAPH_LOC):
            # if the model tar file has not already been downloaded, download it
            if not os.path.exists(os.path.join(MODEL_SAVE_DIR_LOC, MODEL_FILE_NAME)):
                # download the model
                print("downloading model . . .")
                # instantiate a URLopener object, then download the file
                opener = urllib.request.URLopener()
                opener.retrieve(DOWNLOAD_MODEL_FROM_LOC + MODEL_FILE_NAME,
                                os.path.join(MODEL_SAVE_DIR_LOC, MODEL_FILE_NAME))
            # end if

            # unzip the tar to get the frozen inference graph
            print("unzipping model . . .")
            tar_file = tarfile.open(os.path.join(MODEL_SAVE_DIR_LOC, MODEL_FILE_NAME))
            for file in tar_file.getmembers():
                file_name = os.path.basename(file.name)
                if 'frozen_inference_graph.pb' in file_name:
                    tar_file.extract(file, MODEL_SAVE_DIR_LOC)
                # end if
            # end for
        # end if

    except Exception as e:
        print("error downloading or unzipping model: " + str(e))
        return
    # end try


def load_model_in_memory():
    # load the frozen model into memory
    print("loading frozen model into memory . . .")
    detection_graph = tf.Graph()
    try:
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(FROZEN_INFERENCE_GRAPH_LOC, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            # end with
        # end with
    except Exception as e:
        print("error loading the frozen model into memory: " + str(e))
        return
    # end try
    return detection_graph


def load_label_map():
    print("loading label map . . .")
    label_map = label_map_util.load_labelmap(LABEL_MAP_LOC)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def run_inference_for_each_frame(image, graph,sess):
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores','detection_classes'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict


if __name__ == '__main__':
    download_model()
    detection_graph = load_model_in_memory()
    category_index = load_label_map()

    videopath = "challenge_video.mp4"
    cap = cv2.VideoCapture(videopath)
    if cap.isOpened():
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while True:
                    red,frame = cap.read()
                    print("read this frame")
                    output_dict = run_inference_for_each_frame(frame,detection_graph,sess)
                    vis_util.visualize_boxes_and_labels_on_image_array(frame,
                                                                       output_dict['detection_boxes'],
                                                                       output_dict['detection_classes'],
                                                                       output_dict['detection_scores'],
                                                                       category_index,
                                                                       use_normalized_coordinates=True,
                                                                       line_thickness=8)
                    cv2.imshow("detection_result",frame)
                    if cv2.waitKey(1)& 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()