#!/usr/bin/env bash

echo "Downloading pre-trained models ..."
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz
wget http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_11_06_2017.tar.gz
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz

echo "Unzipping downloaded models ..."
tar -xvzf ssd_mobilenet_v1_coco_11_06_2017.tar.gz
tar -xvzf ssd_inception_v2_coco_11_06_2017.tar.gz
tar -xvzf rfcn_resnet101_coco_11_06_2017.tar.gz
tar -xvzf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
tar -xvzf faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz
