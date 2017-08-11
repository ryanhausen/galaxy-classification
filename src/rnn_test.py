#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:29:22 2017

@author: ryanhausen
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from network import SimpleNet, SimpleRNN
from datahelper import DataHelper

x = SimpleNet.x
cnn = SimpleNet.build_graph(x)

rnn = SimpleRNN.build_graph(cnn)
infer = tf.nn.softmax(rnn)

y = SimpleRNN.y

dh = DataHelper()

saver = tf.train.Saver()
#print('Recovering')
#saver.recover_last_checkpoints('../models')

with tf.Session() as sess:
    print('Restoring')
    saver.restore(sess, '../models/cnn-rnn-50.ckpt')

    while True:
        img, lbl = dh.get_next_example()

        _y = sess.run(infer, feed_dict={x:img})
        print(f'Label:{lbl}\nGuess:{_y}')

        for i in range(4):
            plt.figure()
            plt.imshow(img[0,:,:,i], cmap='gray')

        plt.show()