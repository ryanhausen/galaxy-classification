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
learning_rate = 0.003

x = SimpleNet.x
cnn = SimpleNet.build_graph(x)

rnn = SimpleRNN.build_graph(cnn)
y = SimpleRNN.y

infer = tf.nn.softmax(rnn)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=rnn, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=iters)

init = tf.global_variables_initializer()

dh = DataHelper(batch_size=50)

epoch_limit = 30
with tf.Session() as sess:
    sess.run(init)

    print(SimpleNet.print_total_params())

    for epoch in range(1, epoch_limit+1):
        start = time.time()

        tf.logging.info(f'Epoch:{epoch}')
        tf.logging.info(f'Training...')

        total = len(dh._train_imgs)
        while dh.training:
            batch_xs, batch_ys = dh.get_next_batch()
            sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})

            print(f'{round(100 * (dh._idx/total), 2)}% complete', end='\r')

            if (sess.run(iters) % 10)==0:
                evals = evaluate.evaluate(sess, infer, x, y, batch_xs, batch_ys, 'rnn_training')

        tf.logging.info('Testing...')



        batch_xs, batch_ys = dh.get_next_batch()
        evals = evaluate.evaluate(sess, infer, x, y, batch_xs, batch_ys, 'rnn_testing')



        tf.logging.info(f'Epoch {epoch} took {(time.time() - start)} seconds')




