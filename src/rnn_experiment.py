# -*- coding: utf-8 -*-
# modeled after https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

from colorama import init, Fore
init(autoreset=True)
red = lambda s: Fore.RED + s
yellow = lambda s: Fore.YELLOW + s
green = lambda s: Fore.GREEN + s

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import evaluate
from network import SimpleNet, SimpleRNN
from datahelper import DataHelper

tf.logging.set_verbosity('INFO')

iters = tf.Variable(1, trainable=False, name='iter')
learning_rate = 1e-3
batch_size = 100
iter_limit = 10000
seed = None
beta = 0.5
dropout_keep = -1.0
restore_file = None#'../models/cnn-rnn-25000.ckpt'

x = SimpleNet.x
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope('cnn'):
    cnn = SimpleNet.build_cnn(x, keep_prob, global_avg_pooling=True)

rnn = SimpleRNN.build_graph(cnn)

y = SimpleRNN.y

infer = tf.nn.softmax(rnn)

with tf.name_scope('metrics'):
    evaluate.evaluate_tensorboard(rnn,y)

with tf.name_scope('loss'):
    l2_loss = 0.0
    for var in tf.trainable_variables():
        if 'w' in var.name:
            tf.add(l2_loss, tf.nn.l2_loss(var))
    l2_loss = tf.multiply(l2_loss, beta)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rnn, labels=y))
    loss = tf.add(loss, l2_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    grads = optimizer.compute_gradients(loss)#non_nan = [(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), var) for grad, var in grads]
    with tf.name_scope('clipping'):
        grads = [(tf.clip_by_value(grad, -2, 2), var) for grad, var in grads]
    update = optimizer.apply_gradients(grads, global_step=iters)

with tf.name_scope('grads'):
    for grad, var in grads:
        tf.summary.histogram(f"{var.name.split(':')[0]}", grad)

with tf.name_scope('weights'):
    for grad, var in grads:
        tf.summary.histogram(f"{var.name.split(':')[0]}", var)

init = tf.global_variables_initializer()

scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(-1, 1))
valid_img = lambda a: a.sum()>0 and np.isfinite(a).sum()==np.prod(a.shape)
norm = lambda a: scaler.fit_transform(a.reshape(-1,1)).reshape(84,84).astype(np.float32) if valid_img(a) else a
mean_subtraction = lambda a: norm((a-a.mean()))
identity = lambda a: a
b_trans = lambda a: norm(mean_subtraction(a))

dh = DataHelper(batch_size=batch_size, band_transform_func=b_trans)
epoch = 1
len_epoch = len(dh._train_imgs)
print(f'Epoch length = {len_epoch}')

summaries = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    if restore_file:
        saver.restore(sess, restore_file)
    else:
        sess.run(init)

    trainWriter = tf.summary.FileWriter('../report/tf-log/train', graph=sess.graph)
    testWriter = tf.summary.FileWriter('../report/tf-log/test', graph=sess.graph)

    tf.logging.info((green(SimpleNet.print_total_params())))

    while iters.eval() <= iter_limit:
        current_iter = iters.eval()

        if current_iter%5==0:
            tf.logging.info(f'Iter:{iters.eval()}...')

        batch_xs, batch_ys = dh.get_next_batch(iter_based=True, split_channels=True)
        sess.run(update, feed_dict={x:batch_xs, y:batch_ys, keep_prob:dropout_keep})

        if current_iter % 10 == 0:
            #evals = evaluate.evaluate(sess, infer, x, y, batch_xs, batch_ys, '../report/train_progress.csv')
            s = sess.run(summaries, feed_dict={x:batch_xs, y:batch_ys, keep_prob:-1.0})
            trainWriter.add_summary(s, current_iter)

        if current_iter % 50 == 0:
            tf.logging.info(yellow('Testing...'))
            batch_xs, batch_ys = dh.get_next_batch(iter_based=True, force_test=True, split_channels=True)
            #evals = evaluate.evaluate(sess, infer, x, y, batch_xs, batch_ys, '../report/test_progress.csv')

#            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#            run_metadata = tf.RunMetadata()
#            s = sess.run(summaries,
#                         feed_dict={x:batch_xs, y:batch_ys},
#                         options=run_options,
#                         run_metadata=run_metadata)
#            testWriter.add_run_metadata(run_metadata, f'train{current_iter}')

            s = sess.run(summaries, feed_dict={x:batch_xs, y:batch_ys, keep_prob:-1.0})
            testWriter.add_summary(s, current_iter)

            save_path = saver.save(sess, f'../models/cnn-rnn-{current_iter}.ckpt')
            tf.logging.info(green(f'Checkpoint: {save_path}'))


        if (current_iter * batch_size) > (len_epoch * epoch):
            tf.logging.info(red(f'Epoch {epoch} complete.'))
            epoch += 1
    trainWriter.close()
    testWriter.close()
