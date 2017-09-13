from models import object_detection
from config import config
import cv2

model_name = config.models["5"]
net = object_detection.Net(graph_fp='%s/frozen_inference_graph.pb' % model_name,
                           labels_fp='data/label.pbtxt',
                           num_classes=90)
img = 'test_images/1.jpg'
net.predict(img=cv2.imread(img))
