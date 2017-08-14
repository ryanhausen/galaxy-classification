# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:50:28 2016

@author: ryanhausen
"""
import tensorflow as tf


def top_1(yh,ys):
    return tf.reduce_mean(tf.to_float(tf.nn.in_top_k(yh,tf.argmax(ys, 1),1)), name='TOP-1')

def top_2(yh,ys):
    return tf.reduce_mean(tf.to_float(tf.nn.in_top_k(yh,tf.argmax(ys, 1),2)), name='TOP-2')

def rmse(yh, ys):
    return tf.sqrt(tf.reduce_mean(tf.squared_difference(yh,ys)), name='RMSE')

def cross_entropy(yh, ys):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yh, labels=ys), name='CROSS-ENTROPY')

def class_accuracy_part1(yh,ys):
    return tf.argmax(ys, 1), tf.to_float(tf.nn.in_top_k(yh,tf.argmax(ys, 1),1))

def class_accuracy_part2(yscorrect):
    ys, correct = yscorrect
    classes = []

    for i in range(5):
        classes.append('{}/{}'.format(correct[ys==i].sum(), len(correct[ys==i])))

    return classes

def single_class_accuracy(yh, ys, class_idx):
    lbls = tf.argmax(ys,1)
    correct = tf.equal(tf.arg_max(yh,1), lbls)
    class_examples = tf.equal(lbls, class_idx)

    correct_class_examples = tf.reduce_sum(tf.cast(tf.logical_and(correct, class_examples), tf.float32))
    total = tf.reduce_sum(tf.cast(class_examples, tf.float32))

    return tf.divide(correct_class_examples, total)

def evaluate(session, net, x, y, xs, ys, save_to, train=True):
    funcs = [top_1(net, y),
             top_2(net, y),
             rmse(net, ys),
             cross_entropy(net, ys),
             class_accuracy_part1(net, y)]
    outs = []

    for f in funcs:
        if type(f) == tuple:
            outs.append(class_accuracy_part2(session.run(f, feed_dict={x: xs, y: ys})))
        else:
            outs.append(session.run(f, feed_dict={x: xs, y: ys}))

    out_s = [str(o) for o in outs]
    if save_to:
        with open(save_to, 'a') as f:
            f.write(','.join(out_s)+'\n')
    else:
        tf.logging.info('\n'.join(out_s))

    return outs

def save(outs, save_to):
    out_string = ','.join(outs)+'\n'
    if save_to:
        with open(save_to, 'a') as f:
                f.write(out_string)
    else:
        tf.logging.info(out_string)

def evaluate_tensorboard(logit_y,ys):

    yh = tf.nn.softmax(logit_y)
    tf.summary.scalar('top_1', top_1(yh, ys))
    tf.summary.scalar('top_2', top_2(yh, ys))
    tf.summary.scalar('cross_entropy', cross_entropy(logit_y, ys))
    tf.summary.scalar('Spheroid', single_class_accuracy(yh,ys,0))
    tf.summary.scalar('Disk', single_class_accuracy(yh,ys,1))
    tf.summary.scalar('Irregular', single_class_accuracy(yh,ys,2))
    tf.summary.scalar('Point Source', single_class_accuracy(yh,ys,3))
    tf.summary.scalar('Unknown', single_class_accuracy(yh,ys,4))
