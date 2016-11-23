# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:50:28 2016

@author: ryanhausen
"""
import tensorflow as tf


def _top_1(yh,ys):
    return tf.reduce_mean(tf.to_float(tf.nn.in_top_k(yh,ys,1)))

def _top_2(yh,ys):
    return tf.reduce_mean(tf.to_float(tf.nn.in_top_k(yh,ys,2)))

def _rmse(yh, ys):
    return tf.sqrt(tf.reduce_mean(tf.squared_difference(yh,ys)))

def _class_accuracy_part1(yh,ys):
    return ys, tf.to_float(tf.nn.in_top_k(yh,ys,1))

def _class_accuracy_part2(yscorrect):
    ys, correct = yscorrect
    classes = []
    
    for i in range(5):
        classes.append('{}/{}'.format(correct[ys==i].sum(), len(correct[ys==i])))
        
    return classes
    
def evaluate(session, net, x, y, xs, ys, save_to, train=True):
    yh = tf.nn.softmax(net) if train else net
    y_arg = tf.argmax(ys, 1)
    
    funcs = [_top_1(yh,y_arg), _top_2(yh,y_arg), _rmse(yh,ys), _class_accuracy_part1(yh,y_arg)]    
    outs = []
    
    for f in funcs:
        if type(f) == tuple:
            outs.append(_class_accuracy_part2(session.run(f, feed_dict={x: xs, y: ys})))
        else:
            outs.append(session.run(f, feed_dict={x: xs, y: ys}))
    
    out_s = [str(o) for o in outs]
    if save_to:
        with open(save_to, 'a') as f:
            f.write(','.join(out_s)+'\n')
    else:
        print '\n'.join(out_s)

    return outs