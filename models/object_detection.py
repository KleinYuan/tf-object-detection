import sys
import cv2
import numpy as np
import tensorflow as tf
from copy import deepcopy
sys.path.append("..")
import lib.label_map_util
import datetime

'''
x1,y1 ------
|          |
|          |
|          |
--------x2,y2
'''


class Net:
    def __init__(self, graph_fp, labels_fp, num_classes=90, threshold=0.6):
        self.graph_fp = graph_fp
        self.labels_fp = labels_fp
        self.num_classes = num_classes

        self.graph = None
        self.label_map = None
        self.categories = None
        self.category_index = None

        self.bb = None
        self.bb_origin = None
        self.image_tensor = None
        self.boxes = None
        self.scores = None
        self.classes = None
        self.num_detections = None

        self.session = None
        self.threshold = threshold
        self._load_graph()
        self._load_labels()
        self._init_predictor()

    def _load_labels(self):
        self.label_map = lib.label_map_util.load_labelmap(self.labels_fp)
        self.categories = lib.label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = lib.label_map_util.create_category_index(self.categories)

    def _load_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_fp, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def _display(self, filtered_results, processed_img, display_img):
        h, w, _ = processed_img.shape
        h_dis, w_dis, _ = display_img.shape
        ratio_h = h_dis / h
        ratio_w = w_dis / w

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)
        line_type = 2
        offset = 20
        for res in filtered_results:
            y1, x1, y2, x2 = res["bb_o"]
            y1, y2 = y1 * ratio_h, y2 * ratio_h
            x1, x2 = x1 * ratio_w, x2 * ratio_w
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_img, res["class"],
                        (x1 + offset, y1 - offset),
                        font,
                        font_scale,
                        font_color,
                        line_type)
        cv2.imshow('img', display_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    def _init_predictor(self):
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with self.graph.as_default():
            self.session = tf.Session(config=tf_config, graph=self.graph)
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

    def predict(self, img, display_img):
        start = datetime.datetime.now().microsecond * 0.001

        with self.graph.as_default():
            print '[INFO] Read the image ..'

            img_copy = deepcopy(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            height, width, _ = img.shape
            print '[INFO] Shape of this image is -- [heigh: %s, width: %s]' % (height, width)

            image_np_expanded = np.expand_dims(img, axis=0)

            print '[INFO] Detecting objects ...'
            (boxes, scores, classes, num_detections) = self.session.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={
                    self.image_tensor: image_np_expanded
                })

            print '[INFO] Filtering results ...'
            filtered_results = []
            for i in range(0, num_detections):
                score = scores[0][i]
                if score >= self.threshold:
                    y1, x1, y2, x2 = boxes[0][i]
                    y1_o = int(y1 * height)
                    x1_o = int(x1 * width)
                    y2_o = int(y2 * height)
                    x2_o = int(x2 * width)
                    predicted_class = self.category_index[classes[0][i]]['name']
                    filtered_results.append({
                        "score": score,
                        "bb": boxes[0][i],
                        "bb_o": [y1_o, x1_o, y2_o, x2_o],
                        "img_size": [height, width],
                        "class": predicted_class
                    })
                    print '[INFO] %s: %s' % (predicted_class, score)

            # print 'Displaying %s objects against raw images ... ' % num_detections
            self._display(filtered_results, processed_img=img_copy, display_img=display_img)
        end = datetime.datetime.now().microsecond * 0.001
        elapse = end - start
        print '----------------------- FPS: ', 1000.0/elapse
        # You may feel a little bit ugly below and wonder why we don't use "with", but dude, this is a tensorflow bug,
        # and if you don't do this, your machine memory is gonna explode. bang!
        # session.close()
        # del session

    def kill_predictor(self):
        self.session.close()
        self.session = None
