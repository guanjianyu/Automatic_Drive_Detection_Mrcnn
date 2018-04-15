import model as md
import cv2
import numpy as np
import tensorflow as tf
from lane_detection import lane
from utils import visualization_utils as vis_util

if __name__ == '__main__':
    md.download_model()
    detection_graph = md.load_model_in_memory()
    category_index = md.load_label_map()

    video_path = "project_video.mp4"
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('Resetnet_coco.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)),int(cap.get(4))))
    if cap.isOpened():
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while True:
                    red,frame = cap.read()
                    print("read this frame")
                    undist_frame = lane.undistort(frame, lane.mtx, lane.dist)
                    lane_result = lane.lane_detection(undist_frame)

                    output_dict = md.run_inference_for_each_frame(undist_frame, detection_graph, sess)
                    vis_util.visualize_boxes_and_labels_on_image_array(undist_frame,
                                                                       output_dict['detection_boxes'],
                                                                       output_dict['detection_classes'],
                                                                       output_dict['detection_scores'],
                                                                       category_index,
                                                                       use_normalized_coordinates=True,
                                                                       line_thickness=8)
                    result = cv2.addWeighted(undist_frame, 1, lane_result, 0.3, 0)
                    info = np.zeros_like(result)
                    info[5:50, 5:400] = (255, 255, 255)
                    result = cv2.addWeighted(result, 1, info, 0.2, 0)
                    cv2.putText(result, 'Faster Rcnn Resnet101 coco', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 200), 2)
                    video_writer.write(result)
                    cv2.imshow("image", result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                video_writer.release()
                cap.release()
                cv2.destroyAllWindows()(lane.th_h)


