from metrics.IS import *
from glob import glob
import os

BATCH_SIZE = 32

'''
image: numpy array with shape [-1, 3, 32, 32] within [0, 255]
'''

def inception_score(images) :

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    graph = tf.Graph()
    with graph.as_default():
        inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
        tfgan = tf.contrib.gan
        sess = tf.Session()
    # Run images through Inception.
    
    logits = inception_logits(tfgan, inception_images)

    is_mean, is_std = get_inception_score(sess, BATCH_SIZE, images, inception_images, logits, splits=10)

    return is_mean, is_std
