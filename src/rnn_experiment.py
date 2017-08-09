# -*- coding: utf-8 -*-
# modeled after https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

import time

import numpy as np
import tensorflow as tf

import evaluate
from network import SimpleNet, SimpleRNN
from datahelper import DataHelper

tf.logging.set_verbosity('INFO')

iters = tf.Variable(1, trainable=False)
learning_rate = 0.1

x = SimpleNet.x
cnn = SimpleNet.build_graph(x)

rnn = SimpleRNN.build_graph(cnn)
y = SimpleRNN.y

infer = tf.nn.softmax(rnn)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rnn, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=iters)

init = tf.global_variables_initializer()

#https://stackoverflow.com/a/44983523
norm = lambda a: 2*(a - np.max(a))/-np.ptp(a)-1

dh = DataHelper(batch_size=50, band_transform_func=norm)

iter_limit = 1000
with tf.Session() as sess:
    sess.run(init)

    print(SimpleNet.print_total_params())

    while iters.eval() <= iter_limit:
        tf.logging.info(f'Iter:{iters.eval()}...')
        batch_xs, batch_ys = dh.get_next_batch(iter_based=True)
        sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})

        if iters.eval() % 10 == 0:
            evals = evaluate.evaluate(sess, infer, x, y, batch_xs, batch_ys, '../report/train_progress.csv')

        if iters.eval() % 50 == 0:
            tf.logging.info('Testing...')
            batch_xs, batch_ys = dh.get_next_batch(iter_based=True, force_test=True)
            evals = evaluate.evaluate(sess, infer, x, y, batch_xs, batch_ys, '../report/test_progress.csv')

