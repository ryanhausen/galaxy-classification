import tensorflow as tf

batch_norm = tf.layers.batch_normalization
conv2d = tf.layers.conv2d

BATCH_NORM_MOMENTUM = 0.9

def block_conv(*ignore, x=None,
                        filters=None,
                        stride=1,
                        kernel_size=3,
                        padding='same',
                        data_format='channels_first',
                        weight_init=tf.variance_scaling_initializer()):
    if ignore:
        raise ValueError('Only keyword arguments are supported')
    if x is None:
        raise ValueError('Must specify \'x\'')
    if filters is None:
        raise ValueError('Must specify \'filters\'')

    return conv2d(x,
                  filters,
                  kernel_size,
                  strides=stride,
                  padding=padding,
                  kernel_initializer=weight_init,
                  use_bias=None,
                  data_format=data_format,
                  name='block_conv')


def block_op(*ignore, x=None,
                      activation=tf.nn.relu,
                      is_training=None,
                      weight_op=None,
                      data_format='channels_first'):
    if ignore:
        raise ValueError('Only keyword arguments are supported')
    if is_training is None:
        raise ValueError('Must specify \'is_training\'')
    if x is None:
        raise ValueError('Must specify \'x\'')


    x = batch_norm(x,
                   momentum=BATCH_NORM_MOMENTUM,
                   scale=activation==tf.nn.relu,
                   fused=True,
                   axis=1 if data_format=='channels_first' else 3,
                   training=is_training)

    if activation:
        x = activation(x)

    if weight_op:
        x = weight_op(x)

    return x

def block(*ignore, x=None,
                   is_training=None,
                   projection_op=None,
                   resample_op=None,
                   data_format='channels_first'):
    if ignore:
        raise ValueError('Only keyword arguments are supported')
    if is_training is None:
        raise ValueError('Must specify \'is_training\'')
    if x is None:
        raise ValueError('Must specify \'x\'')

    ch_idx = 1 if data_format=='channels_first' else 3
    in_channels = x.shape.as_list()[ch_idx]

    fx = x

    def standard_conv(_x):
        return block_conv(x=_x,filters=in_channels, data_format=data_format)

    init_conv = resample_op if resample_op else standard_conv

    with tf.variable_scope('block_op1'):
        fx = block_op(x=fx, is_training=is_training, weight_op=init_conv, data_format=data_format)

    with tf.variable_scope('block_op2'):
        fx = block_op(x=fx, is_training=is_training, weight_op=standard_conv, data_format=data_format)

    if projection_op:
        with tf.variable_scope('projection_op'):
            x = projection_op(x)

    return x + fx






