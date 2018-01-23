from copy import deepcopy

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

        with tf.variable_scope('in_conv'):
            None

        segment_outs = []
        # encoder
        for s_idx in range(len(block_config)):
            for b_idx in range(block_config[s_idx]):
                with tf.variable_scope('segment{}_block{}'.format(s_idx, b_idx)):
                    print('segment{}_block{}'.format(s_idx, b_idx), x.shape.as_list())
                    x = ResNetMapper.building_block(x,
                                                    b_idx==0 and s_idx>0,
                                                    is_training,
                                                    reuse,
                                                    ResNetMapper.encoding_block_op,
                                                    conv2d)
                    print('segment{}_block{}'.format(s_idx, b_idx), x.shape.as_list())
            segment_outs.append(x)

        print("encoder constructed")

        # decoder
        for s_idx in range(-1, -(len(block_config)+1), -1):
            x = tf.concat([x, segment_outs[s_idx]], 3)
            for b_idx in range(-1, -(block_config[s_idx]+1), -1):
                with tf.variable_scope('segment{}_block{}'.format(s_idx, b_idx)):
                    print('segment{}_block{}'.format(s_idx, b_idx), x.shape.as_list())
                    x = ResNetMapper.building_block(x,
                                                    b_idx==-1 and s_idx>-4,
                                                    is_training,
                                                    reuse,
                                                    ResNetMapper.decoding_block_op,
                                                    conv2d_transpose)
                    print('segment{}_block{}'.format(s_idx, b_idx), x.shape.as_list())


        with tf.variable_scope('out_conv'):
            print('out_conv')
            x = conv2d(x,
                       6,
                       (1,1),
                       biases_initializer=None,
                       weights_initializer=variance_scaling_initializer(),
                       padding=ResNetMapper._PAD_SAME,
                       data_format=ResNetMapper._DATA_FORMAT)
            print(x.shape.as_list())

        return x

    @staticmethod
    def building_block(x, dim_change, is_training, reuse, block_op, project_op):
        ch_idx = 1 if ResNetMapper._DATA_FORMAT=='channels_first' else 3
        in_channels = x.get_shape().as_list()[ch_idx]
        kernel_size = (3, 3)

        if dim_change:
            if block_op==ResNetMapper.encoding_block_op:
                f_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
                out_channels = in_channels * 2
            else:
                f_x = x
                out_channels = in_channels

            pad = ResNetMapper._PAD_VALID
            stride = (2, 2)
        else:
            f_x = x
            out_channels = in_channels
            pad = ResNetMapper._PAD_SAME
            stride = (1, 1)

        with tf.variable_scope('conv_op1', reuse=reuse):
            f_x = block_op(f_x,
                           out_channels,
                           kernel_size,
                           stride,
                           pad,
                           is_training)

        print(f_x.shape.as_list())
        with tf.variable_scope('conv_op2', reuse=reuse):
            f_x = block_op(f_x,
                           out_channels,
                           kernel_size,
                           (1,1),
                           ResNetMapper._PAD_SAME,
                           is_training)
        print(f_x.shape.as_list())

        if dim_change:
            with tf.variable_scope('projection_op', reuse=reuse):
                porjection_kernel = (1, 1)
                x = project_op(x,
                               out_channels,
                               porjection_kernel,
                               stride=(2,2),
                               padding=pad,
                               weights_initializer=variance_scaling_initializer(),
                               biases_initializer=None,
                               data_format=ResNetMapper._DATA_FORMAT)

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

        if c_func==conv2d:
            return c_func(x,
                        f,
                        k,
                        stride=s,
                        padding=p,
                        weights_initializer=variance_scaling_initializer(),
                        biases_initializer=None,
                        activation_fn=None,
                        data_format=ResNetMapper._DATA_FORMAT)
        else:
            cv = c_func(x,
                        f,
                        k,
                        stride=s,
                        padding=p,
                        weights_initializer=variance_scaling_initializer(),
                        biases_initializer=None,
                        activation_fn=None,
                        data_format=ResNetMapper._DATA_FORMAT)


            shp = x.shape.as_list()
            wh = shp[1]*2 if s[0]==2 else shp[1]
            return tf.slice(cv, [0,1,1,0], [-1, wh, wh, f])



            # shp = x.shape.as_list()
            # if s[0]==2:
            #     wh = shp[1]*2
            # else:
            #     wh = shp[1]

            # out_shape = deepcopy(shp)
            # out_shape[0] = -1
            # out_shape[1] = wh
            # out_shape[2] = wh
            # out_shape[3] = f
            # print(wh, out_shape)



            # kernel = tf.get_variable('w',
            #                          shape=[k[0],k[1], f, shp[-1]],
            #                          initializer=variance_scaling_initializer())
            # return c_func(x,
            #               kernel,
            #               out_shape,
            #               [0, s[0], s[1], 0],
            #               padding=p,
            #               data_format=ResNetMapper._DATA_FORMAT)
