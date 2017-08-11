# -*- coding: utf-8 -*-
# modeled after https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

from colorama import init, Fore
init(autoreset=True)
red = lambda s: Fore.RED + s
yellow = lambda s: Fore.YELLOW + s
green = lambda s: Fore.GREEN + s

import numpy as np
import tensorflow as tf

import evaluate
from network import SimpleNet, SimpleRNN
from datahelper import DataHelper

tf.logging.set_verbosity('INFO')

iters = tf.Variable(1, trainable=False)
learning_rate = 1e-6
iter_limit = 750
batch_size = 50
seed = None


x = SimpleNet.x

cnn = SimpleNet.build_cnn(x, reuse=None, raw_out=True)

rnn = SimpleRNN.build_graph(cnn)
y = SimpleRNN.y

infer = tf.nn.softmax(rnn)

loss = 100*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=rnn, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

grads = optimizer.compute_gradients(loss)
#non_nan = [(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), var) for grad, var in grads]
clipped = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in grads]
update = optimizer.apply_gradients(clipped, global_step=iters)

with tf.name_scope('summaries'):
    for grad, var in clipped:
        tf.summary.histogram(f'Gradients/{var.name}', grad)

init = tf.global_variables_initializer()

#https://stackoverflow.com/a/44983523
norm = lambda a: 2*(a - np.max(a))/-np.ptp(a)-1

dh = DataHelper(batch_size=batch_size)
epoch = 1
len_epoch = len(dh._train_imgs)
print(f'Epoch length = {len_epoch}')

summaries = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    summaryWriter = tf.summary.FileWriter('./tf-log', graph=sess.graph)

    SimpleNet.print_total_params()

    while iters.eval() <= iter_limit:
        tf.logging.info(f'Iter:{iters.eval()}...')
        batch_xs, batch_ys = dh.get_next_batch(iter_based=True, split_channels=True)
        sess.run(update, feed_dict={x:batch_xs, y:batch_ys})




        current_iter = iters.eval()
        if current_iter % 10 == 0:
            evals = evaluate.evaluate(sess, infer, x, y, batch_xs, batch_ys, '../report/train_progress.csv')
            s = sess.run(summaries, feed_dict={x:batch_xs, y:batch_ys})
            summaryWriter.add_summary(s, current_iter)

        if current_iter % 50 == 0:
            tf.logging.info(yellow('Testing...'))
            batch_xs, batch_ys = dh.get_next_batch(iter_based=True, force_test=True, split_channels=True)
            evals = evaluate.evaluate(sess, infer, x, y, batch_xs, batch_ys, '../report/test_progress.csv')


            save_path = saver.save(sess, f'../models/cnn-rnn-{current_iter}.ckpt')
            tf.logging.info(green(f'Checkpoint: {save_path}'))


        if (current_iter * batch_size) > (len_epoch * epoch):
            tf.logging.info(red(f'Epoch {epoch} complete.'))
            epoch += 1