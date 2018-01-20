import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, variance_scaling_initializer
from tensorflow.contrib.layers import conv2d, conv2d_transpose


#https://arxiv.org/pdf/1705.06820.pdf
#https://arxiv.org/pdf/1505.04597.pdf
class ResNetMapper(object):
    _DECAY = 0.9
    _DATA_FORMAT = 'channels_first'
    _PAD_VALID = 'VALID'
    _PAD_SAME = 'SAME'

    @staticmethod
    def build_graph(x, block_config, is_training, reuse):
        """
        """
        # TODO add channel increase to 64 in first block some how

        segment_outs = []
        # encoder
        for s_idx in range(len(block_config)):
            for b_idx in range(block_config[s_idx]):
                with tf.variable_scope('segment{}_block{}'.format(s_idx, b_idx)):
                    x = ResNetMapper.building_block(x,
                                                    b_idx==0,
                                                    is_training,
                                                    reuse,
                                                    ResNetMapper.encoding_block_op,
                                                    conv2d)
            segment_outs.append(tf.copy.deepcopy(x))


        # decoder
        for s_idx in range(-1, -(len(block_config)+1), -1):
            tf.concat(x, segment_outs[s_idx])
            for b_idx in range(-1, -(block_config[s_idx]+1), -1):
                with tf.variable_scope('segment{}_block{}'.format(s_idx, b_idx)):
                    x = ResNetMapper.building_block(x,
                                                    b_idx==-1,
                                                    is_training,
                                                    reuse,
                                                    ResNetMapper.decoding_block_op,
                                                    conv2d_transpose)

        # TODO add conv -> 84x84x1 label
        return x


    @staticmethod
    def building_block(x, dim_change, is_training, reuse, block_op, project_op):
        ch_idx = if _DATA_FORMAT=='channels_first' 1 else 3
        in_channels = x.get_shape().as_list()[ch_idx]
        kernel_size = (3, 3)

        if dim_change:
            f_x = f_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
            out_channels = in_channels * 2
            pad = _PAD_VALID
            stride = (2, 2)
        else:
            f_x = x
            out_channels = in_channels
            pad = _PAD_SAME
            stride = (1, 1)

        with tf.variable_scope('conv_op1', reuse=reuse):
            f_x = block_op(f_x,
                           out_channels,
                           kernel_size,
                           stride,
                           pad,
                           is_training)

        with tf.variable_scope('conv_op2', reuse=reuse):
            f_x = block_op(f_x,
                           out_channels,
                           kernel_size,
                           stride,
                           pad,
                           is_training)

        if dim_change:
            with tf.variable_scope('projection_op', reuse=reuse):
                porjection_kernel = (1, 1)
                x = project_op(x,
                               out_channels,
                               porjection_kernel,
                               strides=stride,
                               padding=pad,
                               kernel_initializer=variance_scaling_initializer(),
                               use_bias=False,
                               data_format=_DATA_FORMAT)

        return tf.add(x, f_x)



    @staticmethod
    def encoding_block_op(x, filters, kernel_size, stride, pad, is_training):
        """
            x:              input tensor
            filters:        int, number of filters
            kernel_size:    tuple/list of 2 ints
            stride:         tuple/list of 2 ints
            pad:            SAME or VALID
            is_training:    bool
        """
        return ResNetMapper._block_op(x,
                                      filters,
                                      kernel_size,
                                      stride,
                                      pad,
                                      is_training,
                                      conv2d)

    @staticmethod
    def decoding_block_op(x, filters, kernel_size, stride, pad, is_training):
        """
            x:              input tensor
            filters:        int, number of filters
            kernel_size:    tuple/list of 2 ints
            stride:         tuple/list of 2 ints
            pad:            SAME or VALID
            is_training:    bool
        """
        return ResNetMapper._block_op(x,
                                      filters,
                                      kernel_size,
                                      stride,
                                      pad,
                                      is_training,
                                      conv2d_transpose)

    @staticmethod
    def _block_op(x, f, k, s, p, is_training, c_func):
        x = batch_norm(x,
                       is_training=is_training,
                       decay=ResNetMapper._DECAY,
                       updates_collections=None)

        x = tf.nn.relu(x)

        return c_func(x,
                      f,
                      k,
                      strides=s,
                      padding=p,
                      kernel_initializer=variance_scaling_initializer(),
                      use_bias=False,
                      data_format=_DATA_FORMAT)
