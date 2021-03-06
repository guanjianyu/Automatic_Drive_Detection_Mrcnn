# Automatic_Drive_Detection_Tensorflow_Convnet

This is an implemention of [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) on Python 3 and TensorFlow. The project adopts four different models in Model Zoo to detect cars on road in demo video. 

This project is built based on [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and you can view more detail at [here](https://github.com/tensorflow/models/tree/master/research/object_detection)

[![](result.gif)](https://www.youtube.com/watch?v=Pv2qcNR-PMs)

### The detection result based on differen models are shown in flowing link videos.
* [ssd_mobilenet_v1_coco](https://www.youtube.com/watch?v=_FdxI0RpHbg)
* [faster_rcnn_inception_v2_coco](https://www.youtube.com/watch?v=79PWOKpy6XQ)
* [faster_rcnn_resnet101_coco](https://www.youtube.com/watch?v=ZAY3yhbmrcY)
* [faster_rcnn_resnet101_kitti](https://www.youtube.com/watch?v=dfwRU9bO6Yk)
* [Final result video](https://www.youtube.com/watch?v=Pv2qcNR-PMs).

### Models in this project
| Model name  | Speed (ms) | COCO mAP[^1] | Outputs |
| ------------ | :--------------: | :--------------: | :-------------: |
| [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz) | 30 | 21 | Boxes |
| [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) | 58 | 28 | Boxes |
| [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz) | 106 | 32 | Boxes |
| Model name | Speed (ms) | Pascal mAP@0.5 (ms) | Outputs|
[faster_rcnn_resnet101_kitti](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_kitti_2018_01_28.tar.gz) | 79  | 87              | Boxes

