from collections import namedtuple

import numpy as np
import tensorflow as tf
from astropy.io import fits

#import convnet_semantic_segmentation as network
from convnet_semantic_segmentation import Model
import tf_logger as log
tf.logging.set_verbosity(tf.logging.INFO)

DataSet = namedtuple('DataSet', ['NUM_LABELS'])
d = DataSet(5)

batch_size = 2000
x = tf.placeholder(tf.float32, shape=[None,40,40,1])
y = tf.placeholder(tf.float32, shape=[None,40,40,6])

log.info('Building graph...')
#block_config = [2,4,4,8]
Model.DATA_FORMAT = 'channels_last'
net = Model(d, False).inference(x)
#network.ResNetMapper._DATA_FORMAT = 'NHWC'
#net = network.ResNetMapper.build_graph(x, block_config, False, None)
log.info('Done')

log.info('Grabbing data...')
all_imgs = fits.getdata('../data/candel_slices/slices.fits')
num_imgs = all_imgs.shape[0]
log.info('Done')

saver = tf.train.Saver()

outs = None
model_dir = '../models/run_3'
with tf.Session() as sess:
    log.info('Restoring model...')
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    iters=0
    while(True):
        start = iters * batch_size
        end = min(start + batch_size, num_imgs)
        xs = all_imgs[start:end]

        # Per image standardization
        #xs = xs - xs.mean(axis=(1,2))[:,np.newaxis,np.newaxis]
        #std = xs.std(axis=(1,2))
        #std[std<=0] = 1/np.sqrt(np.prod(xs.shape[1:]))
        #xs = xs / std[:,np.newaxis,np.newaxis]
        if len(xs.shape)==3:
            xs = xs[:,:,:,np.newaxis]

        log.info('From {} to {}'.format(start, end))
        ys = sess.run(net, feed_dict={x:xs})

        fits.PrimaryHDU(ys).writeto('../data/candels_out/classifications-{}.fits'.format(iters))
        if end == num_imgs:
            break

        iters += 1

