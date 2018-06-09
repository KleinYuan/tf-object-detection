# Scope

This repo is aiming to provide production ready **2D object detection** code basics.

It's based on official tensorflow API jupyter notebook but I will gradually add more popular models such as `yolo` series.

### More

If you are interested in **3D object detection**, visit this [repo](https://github.com/KleinYuan/tf-3d-object-detection).

If you are interested in **Segmentation**, visit this [repo](https://github.com/KleinYuan/tf-segmentation).


# Introduction

This is a repo for implementing object detection with pre-trained models (as shown below) on tensorflow.

| Model name  | Speed | COCO mAP | Outputs |
| ------------ | :--------------: | :--------------: | :-------------: |
| [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz) | fast | 21 | Boxes |
| [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz) | fast | 24 | Boxes |
| [rfcn_resnet101_coco](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_11_06_2017.tar.gz)  | medium | 30 | Boxes |
| [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz) | medium | 32 | Boxes |
| [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz) | slow | 37 | Boxes |


Dependencies:

- [X] Tensorflow >= 1.2.0
- [X] OpenCV


# Run Demo


```
# Clone this repo
git clone https://github.com/KleinYuan/tf-object-detection.git

# Setting up
cd tf-object-detection
bash setup.sh

# Run demo
python app.py

```

![res](https://user-images.githubusercontent.com/8921629/32482793-24968e20-c34e-11e7-9810-4aef685d067f.jpg)

# Image Classifications

I also put an image classifications inference app (VGG16) here.

```
# Assuming you already run setup.sh, which will download vgg16.np

python app_ic.py
```

# Networks

| Model name  | Architecture|
| ------------ | :--------------: |
| [AlextNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) | ![AlexNet](https://kratzert.github.io/images/finetune_alexnet/alexnet.png)|
| [Vgg 16](https://arxiv.org/abs/1409.1556) | ![VGG16](https://www.cs.toronto.edu/~frossard/post/vgg16/vgg16.png)|
| [SSD](https://arxiv.org/abs/1512.02325) | ![SSD](http://joshua881228.webfactional.com/media/uploads/ReadingNote/arXiv_SSD/SSD.png)|
| [ResNet](http://arxiv.org/abs/1512.03385)|![Resnet](https://image.slidesharecdn.com/lenettoresnet-170509055515/95/lenet-to-resnet-17-638.jpg)|
| [MobileNet](https://arxiv.org/abs/1704.04861)|![MobileNet](http://machinethink.net/images/mobilenets/Architecture@2x.png) |
| [Faster R-CNN](https://arxiv.org/abs/1506.01497) | ![fasterrcnn](https://raw.githubusercontent.com/sunshineatnoon/Paper-Collection/master/images/faster-rcnn.png)|