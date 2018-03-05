import tensorflow as tf
from tf.layers import batch_normalization


BATCH_NORM_MOMENTUM = 0.9

def block_op(x, is_training):
    x = batch_normalization(x, 
                            momentum=BATCH_NORM_MOMENTUM,
                            scale=False, # turn off becuase of ReLU, see docs
                            fused=True,
                            training=is_training)

    x = tf.nn.relu(x)

    