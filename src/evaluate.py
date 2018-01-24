# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:50:28 2016

@author: ryanhausen
"""
import tensorflow as tf
import numpy as np

import io
import re
#import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
from textwrap import wrap
import itertools
import tfplot

def top_1(yh,ys):
    return tf.reduce_mean(tf.to_float(tf.nn.in_top_k(yh,tf.argmax(ys, 1),1)), name='TOP-1')

def top_2(yh,ys):
    return tf.reduce_mean(tf.to_float(tf.nn.in_top_k(yh,tf.argmax(ys, 1),2)), name='TOP-2')

def rmse(yh, ys):
    return tf.sqrt(tf.reduce_mean(tf.squared_difference(yh,ys)), name='RMSE')

def cross_entropy(yh, ys):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yh, labels=ys), name='CROSS-ENTROPY')

def weighted_cross_entropy(yh, ys):
    loss_weights = tf.subtract(1.0, tf.reduce_max(tf.multiply(0.8, ys), axis=-1))
    return tf.losses.softmax_cross_entropy(ys,
                                           yh,
                                           weights=loss_weights,
                                           reduction=tf.losses.Reduction.MEAN)

def entropy(ys):
    log_y = tf.log(ys)
    mask = tf.logical_or(tf.is_inf(log_y), tf.is_nan(log_y))
    log_y = tf.where(mask, tf.zeros_like(ys), log_y)

    return tf.reduce_sum(tf.multiply(-1.0, tf.multiply(ys, log_y)))

def agreement(ys):
    return 1 - (entropy(ys) / tf.log(5.0))

def accuracy_by_agreement(yh, ys, agreement_bounds):
    agr = agreement(ys)
    acc = top_1(yh, ys)

    bottom = tf.greater(agr, agreement_bounds[0])
    top  = tf.less_equal(agr, agreement_bounds[1])

    mask = tf.logical_and(bottom, top)
    mask = tf.cast(mask, tf.float32)

    return tf.reduce_mean(tf.multiply(mask, acc), name="ACC-BY-AGR")

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
    correct = tf.equal(tf.argmax(yh,1), lbls)
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

# def get_plot_img(ys, yh):
#     c_matrix = confusion_matrix(ys, yh)
#     plt.figure()
#     plt.imshow(c_matrix, interpolation='nearest', cmap='Blues')
#     classes = ['Sph', 'Disk', 'Irr', 'PS', 'Unk']
#     ticks = [0,1,2,3,4]
#     plt.xticks(ticks, classes, rotation=45)
#     plt.yticks(ticks, classes)
#     plt.xlabel('Prediction')
#     plt.ylabel('Label')
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     img = tf.image.decode_png(buf.getvalue(), channels=4)
#     buf.close()

#     img = tf.expand_dims(img, 0)


#     return img

#https://stackoverflow.com/a/48030258
def plot_confusion_matrix(ys, yh, tensor_name='MyFigure/image', normalize=True):
    labels = ['Spheroid', 'Disk', 'Irregular', 'Point Source', 'Unknown', 'Background']

    cm = confusion_matrix(ys, yh)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')


    np.set_printoptions(precision=2)
    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary


def evaluate_tensorboard(logit_y,ys):

    if len(logit_y.shape.as_list()) > 2:
        logit_y = tf.reshape(logit_y, [-1, 6])
        ys = tf.reshape(ys, [-1, 6])

    yh = tf.nn.softmax(logit_y)
    tf.summary.scalar('top_1', top_1(yh, ys))
    tf.summary.scalar('top_2', top_2(yh, ys))
    tf.summary.scalar('cross_entropy', cross_entropy(logit_y, ys))
    tf.summary.scalar('weighted_cross_entropy', weighted_cross_entropy(logit_y, ys))
    tf.summary.scalar('Spheroid', single_class_accuracy(yh,ys,0))
    tf.summary.scalar('Disk', single_class_accuracy(yh,ys,1))
    tf.summary.scalar('Irregular', single_class_accuracy(yh,ys,2))
    tf.summary.scalar('Point_Source', single_class_accuracy(yh,ys,3))
    tf.summary.scalar('Unknown', single_class_accuracy(yh,ys,4))
    tf.summary.scalar('Background', single_class_accuracy(yh,ys,5))

    # bounds = np.linspace(0, 1, 11)
    # for b in zip(bounds[:-1], bounds[1:]):
    #     bot, top = np.round(b[0], decimals=1), np.round(b[1], decimals=1)
    #     tf.summary.scalar(f'Acc_Agr_{bot}_{top}', accuracy_by_agreement(yh, ys, b))

    #c_ys = tf.argmax(ys, 1)
    #c_yh = tf.argmax(yh, 1)
    #plot_confusion_matrix(ys, yh)

    # c_matrix = tf.cast(tf.confusion_matrix(c_ys, c_yh, num_classes=6), tf.float16)
    # tf.summary.image('Confusion_Matrix', tf.reshape(c_matrix, [1, 6, 6, 1]))

def tensorboard_confusion_matrix(logit_y, ys):
    logit_y = logit_y[0]
    ys = ys.reshape([-1, 6])

    c_ys = np.argmax(ys, axis=1)
    c_yh = np.argmax(logit_y, axis=1)
    return plot_confusion_matrix(c_ys, c_yh)
