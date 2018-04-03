import tensorflow as tf
import numpy as np
from models import vgg16
from services import data
from config import config


img = data.load_image('test_images/2.jpg')
batch = img.reshape((1, 224, 224, 3))

with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder('float', [1, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16(config.ic_models['vgg16'])
        with tf.name_scope('content_vgg'):
            vgg.build(images)

        res = sess.run(vgg.prob, feed_dict=feed_dict)
        prob = res[0]
        pred = np.argsort(prob)[::-1]
        synset = [l.strip() for l in open('data/synset.txt').readlines()]

        prediction = synset[pred[0]]
        confidence = prob[pred[0]]
        print('Prediction is %s\nConfidence is %s' %(prediction, confidence))
