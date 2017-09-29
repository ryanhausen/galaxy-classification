#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:46:34 2017

@author: ryanhausen
"""
import Edward as ed
from Edward.models import Normal, Categorical
from observations import mnist
import Tensorflow as tf

import numpy as np

batch_size = 128

(x_train, y_train), (x_test, y_test) = mnist("~/data")

normal = lambda shp, name: Normal(loc=tf.zeros(shp), scale=tf.ones(shp), name=name)
rand_norm = lambda shp, name: tf.Variable(tf.random_normal(shp), name=name)
qnormal = lambda shp: Normal(loc=rand_norm(shp,'loc'), scale=tf.nn.softplus(rand_norm(shp, 'scale')))

def bnn(x):
    stride1 = [1, 1, 1, 1]
    stride2 = [1, 2, 2, 1]
    valid, same = 'VALID', 'SAME'

    # conv1
    x = tf.nn.conv2d(x, W0, s=stride1, pad=valid) + B0
    x = tf.nn.relu(x)

    # conv2
    x = tf.nn.conv2d(x, W1, s=stride1, pad=valid) + B1
    x = tf.nn.relu(x)

    # reshape
    flat_dim = tf.cumprod(x.shape.as_list())
    x = tf.rehsape(x, [-1, flat_dim])

    # fc
    x = tf.matmul(x, W2) + B2
    x = tf.nn.relu(x)

    # out
    return tf.matmul(x, W3) + B3



with tf.name_scope('model'):
    W0 = normal([5,5,1,32], 'W0')
    W1 = normal([3,3,32,64], 'W1')
    W2 = normal([np.prod([3,3,32,64]), 1024], 'W2')
    W3 = normal([1024, 10], 'W3')
    B0 = normal([32], 'B0')
    B1 = normal([64], 'B1')
    B2 = normal([1024], 'B2')
    B3 = normal([10], 'B3')

    x = tf.placeholder(tf.float32, [batch_size, 28, 28, 1])
    yh = Categorical(logits=bnn(x), name='yh')


with tf.name_scope('posterior'):
    with tf.name_scope('qW0'):
        qW0 = qnormal([5,5,1,32])
    with tf.name_scope('qW1'):
        qW1 = qnormal([3,3,32,64])
    with tf.name_scope('qW2'):
        qW2 = qnormal([np.prod([3,3,32,64]), 1024])
    with tf.name_scope('qW3'):
        qW3 = qnormal([1024,10])
    with tf.name_scope('qB0'):
        qB0 = qnormal([32])
    with tf.name_scope('qB1'):
        qB1 = qnormal([64])
    with tf.name_scope('qB2'):
        qB2 = qnormal([1024])
    with tf.name_scope('qB3'):
        qB3 = qnormal([10])

ws = [W0,W1,W2,W3,B0,B1,B2,B3]
qs = [qW0,qW1,qW2,qW3,qB0,qB1,qB2,qB3]
qmap = {q:p for q,p in zip(ws,qs)

inference = ed.KLqp(qmap, data={X:x_train, y:y_train}
inference.run()
