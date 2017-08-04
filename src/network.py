import numpy as np
import math
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.rnn import BasicRNNCell, static_rnn

class SimpleRNN:
    num_channels = 4
    num_hidden = 32
    out_classes = 5
    x = tf.placeholder(tf.float32, [None,num_channels,out_classes])
    y = tf.placeholder(tf.float32, [None,out_classes])

    @staticmethod
    #https://arxiv.org/abs/1502.01852
    def var_init(shape, dtype, partition_info=None):
        std = None

        # bias
        if len(shape) ==1:
            return tf.constant(0.01, shape=shape)
        # fc Layer
        if len(shape) == 2:
            std = math.sqrt(1/shape[0])
        # conv layer
        elif len(shape) == 4:
            k_sqr_d = shape[0] * shape[1] * shape[2]
            std = math.sqrt(2 / k_sqr_d)

        return tf.truncated_normal(shape, stddev=std, seed=SimpleNet.seed)


    def build_graph(x) :

       # reformat for the RNN
       x = tf.unstack(x, SimpleRNN.num_channels, 1)

       # rnn
       with tf.variable_scope('rnn'):
           rnn = BasicRNNCell(SimpleRNN.num_hidden)

       # process rnn
       outputs, states = static_rnn(rnn, x, dtype=tf.float32)


       with tf.variable_scope('rnn_act'):
           w = tf.get_variable('w', shape=[SimpleRNN.num_hidden, SimpleRNN.out_classes], initializer=SimpleRNN.var_init)
           b = tf.get_variable('b', shape=[SimpleRNN.out_classes], initializer=SimpleRNN.var_init)

       return tf.nn.bias_add(tf.matmul(outputs[-1], w), b)




class SimpleNet:
    num_channels = 4

    seed = None
    x = tf.placeholder(tf.float32, [None,84,84,4])
    y = tf.placeholder(tf.float32, [None, 5])

    @staticmethod
    #https://arxiv.org/abs/1502.01852
    def var_init(shape, dtype, partition_info=None):
        std = None

        # bias
        if len(shape) ==1:
            return tf.constant(0.01, shape=shape)
        # fc Layer
        if len(shape) == 2:
            std = math.sqrt(1/shape[0])
        # conv layer
        elif len(shape) == 4:
            k_sqr_d = shape[0] * shape[1] * shape[2]
            std = math.sqrt(2 / k_sqr_d)

        return tf.truncated_normal(shape, stddev=std, seed=SimpleNet.seed)

    @staticmethod
    def conv2d(x, k, pad='SAME'):
        w = tf.get_variable('w', shape=k, initializer=SimpleNet.var_init)
        b = tf.get_variable('b', shape=[k[-1]], initializer=SimpleNet.var_init)

        strides = [1,1,1,1]

        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w, strides, pad), b))

    @staticmethod
    def max_pool(x, k=2):
        ks = [1, k, k, 1]
        return tf.nn.max_pool(x, ksize=ks, strides=ks, padding='VALID')


    @staticmethod
    def fc(x, n, act=None):
        shp = x.get_shape().as_list()[1]
        w = tf.get_variable('w', shape=[shp, n], initializer=SimpleNet.var_init)
        b = tf.get_variable('b', shape=[n], initializer=SimpleNet.var_init)

        if act:
            return act(tf.nn.bias_add(tf.matmul(x, w), b))
        else:
            return tf.nn.bias_add(tf.matmul(x, w), b)

    @staticmethod
    def c1w_shape():
        return [3,3,1,32]

    @staticmethod
    def c2w_shape():
        return [5, 5, 32, 64]

    @staticmethod
    def c3w_shape():
        return [3, 3, 64, 128]

    @staticmethod
    def c4w_shape():
        return [3, 3, 128, 128]

    @staticmethod
    def f1w_shape():
        return [6272, 2048]

    @staticmethod
    def f2w_shape():
        return [2048, 2048]

    @staticmethod
    def f3w_shape():
        return [2048, 5]

    @staticmethod
    def build_cnn(x, reuse):
        with tf.variable_scope('conv1', reuse=reuse):
            x = SimpleNet.conv2d(x, SimpleNet.c1w_shape())

        x = SimpleNet.max_pool(x)

        with tf.variable_scope('conv2', reuse=reuse):
            x = SimpleNet.conv2d(x, SimpleNet.c2w_shape())

        x = SimpleNet.max_pool(x)

        with tf.variable_scope('conv3', reuse=reuse):
            x = SimpleNet.conv2d(x, SimpleNet.c3w_shape())

        with tf.variable_scope('conv4', reuse=reuse):
            x = SimpleNet.conv2d(x, SimpleNet.c4w_shape())

        x = SimpleNet.max_pool(x)

        x_flat = np.prod(x.get_shape().as_list()[1:])

        x = tf.reshape(x, [-1, x_flat])

        with tf.variable_scope('fc1', reuse=reuse):
            x = SimpleNet.fc(x, SimpleNet.f1w_shape()[1], tf.nn.relu)

        x = tf.nn.dropout(x, 0.5)

        with tf.variable_scope('fc2', reuse=reuse):
            x = SimpleNet.fc(x, SimpleNet.f2w_shape()[1])

        x = tf.nn.dropout(x, 0.5)

        with tf.variable_scope('fc3', reuse=reuse):
            x = SimpleNet.fc(x, SimpleNet.f3w_shape()[1])

        return x

    @staticmethod
    def build_graph(x):

        #break up channels for individaul processing
        # x.shape = [batch_size, 84, 84, num_channels] => list([batch_size, 84, 84, 1])
        # with a length of num_channels
        x = tf.split(x, SimpleNet.num_channels, 3)

        ys = []
        for channel in range(SimpleNet.num_channels):
            ys.append(SimpleNet.build_cnn(x[channel], reuse=channel>0))

        return tf.stack(ys, axis=1)




class CandleNet:
    @staticmethod
    def get_network(x):
        """x should be tf.placeholder(tf.float32, [batch_size, 45, 45, 3])"""
        n_classes = 5
        batch_size = x.get_shape().as_list()[0]
        channels = x.get_shape().as_list()[3]

        # Model Helpers --------------------------------------------------------

        # https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#conv2d
        def conv2d(img, w, b):

            x = tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='VALID')
            z = tf.nn.bias_add(x, b)
            return tf.nn.relu(z)

        # https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#max_pool
        def max_pool(img, k):
            ks = [1, k, k, 1]
            return tf.nn.max_pool(img, ksize=ks, strides=ks, padding='VALID')

        # TODO implement
        def maxout(x):
            raise NotImplemented()

        def fc(x, w, b, act):
            if act:
                return act(tf.add(tf.matmul(x, w), b))
            else:
                return tf.add(tf.matmul(x, w), b)

        def conv_net(_X, _weights, _biases):
            # First convolution layer
            #print 'x: {}'.format(_X.get_shape())

            conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
            # k used to be 2
            conv1 = max_pool(conv1, k=2)

            #print 'conv1: {}'.format(conv1.get_shape())

            # Second Covolution layer
            conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
            conv2 = max_pool(conv2, k=2)

            #print 'conv2: {}'.format(conv2.get_shape())

            # Thrid Convolution Layer
            conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'])

            #print 'conv3: {}'.format(conv3.get_shape())

            # Fourth Convolution Layer
            conv4 = conv2d(conv3, _weights['wc4'], _biases['bc4'])
            conv4 = max_pool(conv4, k=2)

            #print 'conv4: {}'.format(conv4.get_shape())

            # In the paper the FC layers suggest that you use maxout, but
            # there isn't a native maxout in TensorFlow, so I used ReLU for now.

            # First Fully Connected Layer, flatten out filters first
            fc1 = tf.reshape(conv4, [batch_size, -1])
            # https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#relu
            fc1 = fc(fc1, _weights['wf1'], _biases['bf1'], tf.nn.relu)
            # TODO dropout should be a parameter
            fc1  = tf.nn.dropout(fc1, tf.Variable(tf.constant(0.5)))


            # Second Fully Connected Layer
            fc2 = fc(fc1, _weights['wf2'], _biases['bf2'], tf.nn.relu)
            # TODO dropout should be a parameter
            fc2  = tf.nn.dropout(fc2, tf.Variable(tf.constant(0.5)))

            # Output
            # https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#sigmoid
            output = fc(fc2, _weights['out'], _biases['out'], None)
            return output

        # Model Helpers --------------------------------------------------------


        # Model weights and biases
        weights = {
            # 6x6 conv, 3-channel input, 32-channel outputs
            'wc1': tf.Variable(tf.truncated_normal([3, 3, channels, 32], stddev=0.01)), #0.01
            # 5x5 conv, 32-channel inputs, 64-channel outputs
            'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01)), #0.01
            # 3x3 conv, 64-channel inputs, 128-channel outputs
            'wc3': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.01)), #0.01
            # 3x3 conv, 128-channel inputs, 128-channel outputs
            'wc4': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)), #0.1
            # fully connected, 512 inputs, 2048 outputs
            # was 4608 for 84x84
            'wf1': tf.Variable(tf.truncated_normal([6272, 2048], stddev=0.001)), #0.001
            # fully coneected 2048 inputs, 2048 outputs
            'wf2': tf.Variable(tf.truncated_normal([2048, 2048], stddev=0.001)), #0.001
            # 2048 inputs, 5 outputs (class prediction)
            'out': tf.Variable(tf.truncated_normal([2048, n_classes], stddev=0.01)) #0.01
        }

        biases = {
            'bc1': tf.Variable(tf.constant(0.1, shape=[32])),
            'bc2': tf.Variable(tf.constant(0.1, shape=[64])),
            'bc3': tf.Variable(tf.constant(0.1, shape=[128])),
            'bc4': tf.Variable(tf.constant(0.1, shape=[128])),
            'bf1': tf.Variable(tf.constant(0.01, shape=[2048])),
            'bf2': tf.Variable(tf.constant(0.01, shape=[2048])),
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
        }

        return conv_net(x, weights, biases)

class ExperimentalNet:
    @staticmethod
    def get_network(x):
        """x should be tf.placeholder(tf.float32, [batch_size, w, h, channels])"""
        n_classes = 5
        batch_size = x.get_shape().as_list()[0]
        channels = x.get_shape().as_list()[3]

        # split channels to process separately
        c1, c2, c3, c4 = tf.split(3, channels, x)

        # Model Helpers --------------------------------------------------------

        # https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#conv2d
        def conv2d(img, w, b):

            x = tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='VALID')
            z = tf.nn.bias_add(x, b)
            return tf.nn.relu(z)

        # https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#max_pool
        def max_pool(img, k):
            ks = [1, k, k, 1]
            return tf.nn.max_pool(img, ksize=ks, strides=ks, padding='VALID')

        # TODO implement
        def maxout(x):
            raise NotImplemented()

        def fc(x, w, b, act):
            return act(tf.add(tf.matmul(x, w), b))

        def conv_net(_x):
            # First convolution layer
            #print 'x: {}'.format(_X.get_shape())
            weights = {
            # 6x6 conv, 3-channel input, 32-channel outputs
            'wc1': tf.Variable(tf.truncated_normal([10, 10, 1, 32], stddev=0.01)),
            # 5x5 conv, 32-channel inputs, 64-channel outputs
            'wc2': tf.Variable(tf.truncated_normal([7, 7, 32, 64], stddev=0.01)),
            # 3x3 conv, 64-channel inputs, 128-channel outputs
            'wc3': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.01)),
            # 3x3 conv, 128-channel inputs, 128-channel outputs
            'wc4': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)),
            }

            biases = {
            'bc1': tf.Variable(tf.constant(0.1, shape=[32])),
            'bc2': tf.Variable(tf.constant(0.1, shape=[64])),
            'bc3': tf.Variable(tf.constant(0.1, shape=[128])),
            'bc4': tf.Variable(tf.constant(0.1, shape=[128])),
            }


            conv1 = conv2d(_x, weights['wc1'], biases['bc1'])
            # k used to be 2
            conv1 = max_pool(conv1, k=4)

            # Second Covolution layer
            conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
            conv2 = max_pool(conv2, k=2)

            # Thrid Convolution Layer
            conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])

            #print 'conv3: {}'.format(conv3.get_shape())

            # Fourth Convolution Layer
            conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
            conv4 = max_pool(conv4, k=2)

            return tf.reshape(conv4, [batch_size, -1])

        fc_weights = {
            'wf1': tf.Variable(tf.truncated_normal([512, 2048], stddev=0.001)),
            # fully coneected 2048 inputs, 2048 outputs
            'wf2': tf.Variable(tf.truncated_normal([2048, 2048], stddev=0.001)),
            # 2048 inputs, 5 outputs (class prediction)
            'out': tf.Variable(tf.truncated_normal([2048, n_classes], stddev=0.01))
        }

        fc_biases = {
            'bf1': tf.Variable(tf.constant(0.01, shape=[2048])),
            'bf2': tf.Variable(tf.constant(0.01, shape=[2048])),
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
        }

        c1 = conv_net(c1)
        c2 = conv_net(c2)
        c3 = conv_net(c3)
        c4 = conv_net(c4)

        # feed this into one fully connected layer
        cmb = tf.concat(1, [c1,c2,c3,c4])

        # fully connected
        fc1 = fc(cmb, fc_weights['wf1'], fc_biases['bf1'], tf.nn.relu)
        fc2 = fc(fc1, fc_weights['wf2'], fc_biases['bf2'], tf.nn.relu)

        # output
        output = fc(fc2, fc_weights['out'], fc_biases['out'], tf.nn.softmax)

        return output

# https://arxiv.org/abs/1512.03385
# https://arxiv.org/abs/1603.05027
class ResNet:
    # class helper variables
    random_seed = None
    X = tf.placeholder(tf.float32, [None, 84, 84, 4])
    Y = tf.placeholder(tf.float32, [None, 5])

    #https://stackoverflow.com/a/38161314
    @staticmethod
    def print_total_params():
        """
            outputs the number of learnable parameters
        """

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        tf.logging.info(f'TOTAL NUMBER OF PARAMTERS:{total_parameters}')

    @staticmethod
    def build_graph(x, block_config, is_training):
        """
            x: a tf.placeholder
            block_config: list of ints that determines the number of residual
                          blocks in each block_segment
            is_training: indicates training vs inference
        """
        # if the network is large use the bottleneck architecture
        is_large = sum(block_config) * 2 >= 50
        reuse = None if is_training else True

        # this initial convolution reduces our image WxH by half
        with tf.variable_scope('conv1', reuse=reuse):
            x = ResNet.conv1(x)

        # begin residual blocks
        for segment_idx in range(len(block_config)):
            num_blocks = block_config[segment_idx]

            for block_idx in range(num_blocks):
                tf.logging.info(f'SEG{segment_idx}-BLOCK{block_idx}')
                with tf.variable_scope(f'seg{segment_idx}block{block_idx}', reuse=reuse):
                    # if this is the first block the WxH are reduced and  the number
                    # of filters is increased
                    reduce = block_idx == 0
                    first = segment_idx==0
                    if is_large:
                        x = ResNet.bottleneck(x, reduce, first, is_training)
                    else:
                        x = ResNet.block(x, reduce, first, is_training)

        with tf.variable_scope('out', reuse=reuse):
            x = ResNet.global_average_pooling(x)

        ResNet.print_total_params()

        if is_training:
            return x
        else:
            return tf.nn.softmax(x)

    # https://arxiv.org/pdf/1312.4400.pdf
    @staticmethod
    def global_average_pooling(x):
        """
            implements global average pooling
        """
        tf.logging.info('GLOBAL-AVG-POOLING--------------')
        tf.logging.info(f'X-IN:{x.get_shape().as_list()}')

        in_channels = x.get_shape().as_list()[-1]

        k = [1, 1, in_channels, 5]
        w = tf.get_variable('w', shape=k, initializer=ResNet.var_init)

        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')
        x = tf.reduce_mean(x, [1, 2], name='global_average')

        tf.logging.info(f'X-OUT:{x.get_shape().as_list()}')
        tf.logging.info('--------------------------------')
        return x

    @staticmethod
    def bottleneck(x, reduce, first_block, is_training):
        """
            implements a bottleneck residual block which consists of a
            1x1 convolution -> 3x3 convolution -> 1x1 convolution -> shortcut
        """
        reuse = None if is_training else True
        in_channels = x.get_shape().as_list()[-1]

        shp = lambda t: t.get_shape().as_list()
        tf.logging.info('BOTTLENECK:---------------------')
        tf.logging.info(f'X-IN:{shp(x)}')

        if reduce:
            tf.logging.info('Reducing Image Size')

            if first_block:
                bottleneck_channels = in_channels
                stride = 1
                pad = 'SAME'
            else:
                bottleneck_channels = in_channels // 2
                stride = 2
                pad = 'VALID'

            out_channels = in_channels * 2
        else:
            if first_block:
                bottleneck_channels = in_channels // 2
            else:
                bottleneck_channels = in_channels // 4

            out_channels = in_channels
            pad = 'SAME'
            stride = 1

        # 1x1 conv1
        with tf.variable_scope('op1', reuse=reuse):
            k = [1, 1, in_channels, bottleneck_channels]
            f_x = ResNet.block_op(x, k, 1, 'VALID', is_training)

        tf.logging.debug(f'H(X):{shp(f_x)}')

        # 3x3 conv2
        with tf.variable_scope('op2', reuse=reuse):
            if reduce and first_block == False:
                f_x = tf.pad(f_x, [[0, 0], [1, 1], [1, 1], [0, 0]])

            k = [3, 3, bottleneck_channels, bottleneck_channels]
            f_x = ResNet.block_op(f_x, k, stride, pad, is_training)

        tf.logging.debug(f'H(X):{shp(f_x)}')


        # 1x1 conv3
        with tf.variable_scope('op3', reuse=reuse):
            k = [1, 1, bottleneck_channels, out_channels]
            f_x = ResNet.block_op(f_x, k, 1, 'VALID', is_training)

        tf.logging.debug(f'H(X):{shp(f_x)}')

        if reduce:
            k = [1, 1, in_channels, out_channels]
            w = tf.get_variable('proj', shape=k, initializer=ResNet.var_init)
            x = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding='VALID')


        tf.logging.debug(f'X+H(X):{shp(f_x)} + {shp(x)}')
        tf.logging.info(f'X-OUT:{shp(x)}')
        tf.logging.info('--------------------------------')
        return tf.add(x, f_x)

    @staticmethod
    def block(x, reduce, first_block, is_training):
        """
            implements a standard residual block which consists of a
            3x3 convolution -> 3x3 convolution -> shortcut
        """
        in_channels = x.get_shape().as_list()[-1]
        reuse = None if is_training else True

        shp = lambda t: t.get_shape().as_list()
        tf.logging.info('STANDARD:-----------------------')
        tf.logging.info(f'X-IN:{shp(x)}')

        if reduce and first_block==False:
            tf.logging.info('Reducing Image Size')
            out_channels = in_channels * 2
            pad = 'VALID'
            stride = 2
        else:
            out_channels = in_channels
            pad = 'SAME'
            stride = 1

        # 3x3 conv1
        with tf.variable_scope('op1', reuse=reuse):
            if reduce and first_block == False:
                f_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
                k = [3, 3, in_channels, out_channels]
                f_x = ResNet.block_op(f_x, k, stride, pad, is_training)
            else:
                k = [3, 3, in_channels, out_channels]
                f_x = ResNet.block_op(x, k, stride, pad, is_training)

        tf.logging.debug(f'H(X):{shp(f_x)}')

        # 3x3 conv2
        with tf.variable_scope('op2', reuse=reuse):
            k = [3, 3, out_channels, out_channels]
            f_x = ResNet.block_op(f_x, k, 1, 'SAME', is_training)

        tf.logging.debug(f'H(X):{shp(f_x)}')

        # project x into the correct dimensions if there was a reduction
        if reduce and first_block==False:
            k = [1, 1, in_channels, out_channels]
            w = tf.get_variable('proj', k, initializer=ResNet.var_init)
            x = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding=pad)

        tf.logging.debug(f'X+H(X):{shp(f_x)} + {shp(x)}')
        tf.logging.info(f'X-OUT:{shp(x)}')
        tf.logging.info('--------------------------------')
        return tf.add(x, f_x)

    @staticmethod
    def block_op(x, k, s, pad, is_training):
        """
            implements the set of operations that accompany each
            convolution. batchnorm -> relu -> conv2d
        """
        x = batch_norm(x, is_training=is_training, decay=0.9, updates_collections=None)
        x = tf.nn.relu(x)

        w = tf.get_variable('w', shape=k, initializer=ResNet.var_init)
        x = tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding=pad)

        return x

    @staticmethod
    def conv1(x):
        """
           Simple convolutional layer
           pad(3x3) -> conv2d -> relu
        """
        tf.logging.info('INITAL-CONV---------------------')
        tf.logging.info(f'X-IN:{x.get_shape().as_list()}')
        #              Batch   Width   Height  Channels
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'CONSTANT')

        w = tf.get_variable('w', shape=[7, 7, 4, 64], initializer=ResNet.var_init)
        s = [1, 2, 2, 1]

        x = tf.nn.relu(tf.nn.conv2d(x, w, s, 'VALID'))

        tf.logging.info(f'X-OUT:{x.get_shape().as_list()}')
        tf.logging.info('--------------------------------')
        return x


    #https://arxiv.org/abs/1502.01852
    @staticmethod
    def var_init(shape, dtype, partition_info=None):
        std = None

        # bias
        if len(shape) ==1:
            return tf.constant(0.01, shape=shape)
        # fc Layer
        if len(shape) == 2:
            std = math.sqrt(1/shape[0])
        # conv layer
        elif len(shape) == 4:
            k_sqr_d = shape[0] * shape[1] * shape[2]
            std = math.sqrt(2 / k_sqr_d)

        return tf.truncated_normal(shape, stddev=std, seed=ResNet.random_seed)

class Resnet:
    @staticmethod
    def get_network(x, block_config,
                    is_training=True,
                    global_avg_pool=False,
                    use_2016_update=False):
        n_classes = 5

        shp = x.get_shape().as_list()
        batch_size = shp[0]
        channels = shp[3]

        in_dim = 16

        with tf.variable_scope('conv1'):
            weights = Resnet._make_weights([3,3,channels,in_dim], 'weights')
            x = Resnet._block_operations(x,
                                         weights,
                                         s=2,
                                         pad='VALID',
                                         activation=tf.nn.relu)

        block_seg = 0
        for block in block_config:
            block_seg += 1
            for i in range(1, block+1):
                scope = 'Seg{}block{}'.format(block_seg, i)
                with tf.variable_scope(scope):

                    weights = []
                    first = block_seg == 1 and i ==1
                    increase_dim = i == block or first

                    if first:
                        weights = [
                            Resnet._make_weights([1,1,in_dim,in_dim*2], 'w1'),
                            Resnet._make_weights([3,3,in_dim*2,in_dim*2], 'w2'),
                            Resnet._make_weights([1,1,in_dim*2,in_dim*4], 'w3'),
                            Resnet._make_weights([1,1,in_dim,in_dim*4], 'w4')
                            ]
                    else:
                        weights.append(Resnet._make_weights([1,1,in_dim,in_dim/2], 'w1'))

                        if increase_dim:
                            weights.append(Resnet._make_weights([3,3,in_dim/2,in_dim], 'w2'))
                            weights.append(Resnet._make_weights([1,1,in_dim,in_dim*2], 'w3'))
                            weights.append(Resnet._make_weights([3,3,in_dim,in_dim*2], 'w4'))
                        else:
                            weights.append(Resnet._make_weights([3,3,in_dim/2,in_dim/2], 'w2'))
                            weights.append(Resnet._make_weights([1,1,in_dim/2,in_dim], 'w3'))

                    if use_2016_update:
                        x = Resnet._building_block_2016(x,
                                                        weights,
                                                        increase_dim=increase_dim,
                                                        first=first)
                    else:
                        x = Resnet._building_block(x,
                                                   weights,
                                                   increase_dim=increase_dim,
                                                   first=first)

                    if increase_dim:
                        in_dim *= 2
                    if first:
                        in_dim *= 2


        output = None

        # conclude the network with global average pooling
        if global_avg_pool:
            with tf.variable_scope('out'):
                w = Resnet._make_weights([1,1,in_dim,n_classes], 'weights')
                x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID')

                shp = x.get_shape().as_list()

                x = Resnet._avg_pool(x, shp[1])
                # avg_pool returns the shape [batch_size,1,1,n_classes]
                # reshape to make it compatible with the label
                output = tf.reshape(x, [-1, 5])

        # conclude the network with fully connected layers
        else:
            fc_transform = tf.reshape(x, [batch_size, -1])


            # fully connected
            transformed_dim = fc_transform.get_shape().as_list()[1]
            fc_weights = {
                'wf1': Resnet._make_weights([transformed_dim, 2048], 'wf1'),
                # fully coneected 2048 inputs, 2048 outputs
                'wf2': Resnet._make_weights([2048, 2048], 'wf2'),
                # 2048 inputs, 5 outputs (class prediction)
                'out': Resnet._make_weights([2048, n_classes], 'out')
            }

            fc_biases = {
                'bf1': tf.Variable(tf.constant(0.01, shape=[2048])),
                'bf2': tf.Variable(tf.constant(0.01, shape=[2048])),
                'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
            }

            fc1 = Resnet._fc(fc_transform, fc_weights['wf1'], fc_biases['bf1'], tf.nn.relu)
            fc2 = Resnet._fc(fc1, fc_weights['wf2'], fc_biases['bf2'], tf.nn.relu)

            # output
            if is_training:
                output = Resnet._fc(fc2, fc_weights['out'], fc_biases['out'], None)
            else:
                output = Resnet._fc(fc2, fc_weights['out'], fc_biases['out'], tf.nn.softmax)

        return output

    @staticmethod
    def _building_block(x, ws,
                        activation=tf.nn.relu,
                        increase_dim=False,
                        first=False,
                        is_training=True):
        dim_stride = 1
        pad = 'SAME'

        if increase_dim and not first:
            dim_stride = 2
            pad = 'VALID'

        # first 1x1 conv
        f_x = Resnet._block_operations(x, ws[0], s=1, pad='VALID', activation=activation, is_training=is_training)

        # 3x3 conv
        f_x = Resnet._block_operations(f_x, ws[1], s=dim_stride, pad=pad, activation=activation, is_training=is_training)

        # second 1x1 conv
        f_x = Resnet._block_operations(f_x, ws[2], s=1, pad='VALID', activation=None, is_training=is_training)

        if increase_dim:
            x = Resnet._block_operations(x, ws[3], s=dim_stride, pad='VALID', activation=None, is_training=is_training)

        x = x + f_x

        return activation(x)

    @staticmethod
    def _block_operations(x, w,
                          b=None,
                          s=1,
                          pad='SAME',
                          batch_norm=True,
                          is_training=True,
                          activation=None):

        x = tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding=pad)

        if batch_norm:
            x = Resnet._batch_norm_layer(x, is_training=is_training)
        else:
            x = tf.nn.bias_add(x, b)

        if activation:
            x = activation(x)

        return x

    @staticmethod
    def _building_block_2016(x, ws,
                            activation=tf.nn.relu,
                            increase_dim=False,
                            first=False,
                            is_training=True):
        dim_stride = 1
        pad = 'SAME'

        if increase_dim and not first:
            dim_stride = 2
            pad = 'VALID'

        # first 1x1 conv
        f_x = Resnet._block_operations(x, ws[0], s=1, pad='VALID', activation=activation, is_training=is_training)

        # 3x3 conv
        f_x = Resnet._block_operations(f_x, ws[1], s=dim_stride, pad=pad, activation=activation, is_training=is_training)

        # second 1x1 conv
        f_x = Resnet._block_operations(f_x, ws[2], s=1, pad='VALID', activation=activation, is_training=is_training)

        if increase_dim:
            x = Resnet._block_operations(x, ws[3], s=dim_stride, pad='VALID', activation=None, is_training=is_training)

        return x + f_x

    @staticmethod
    def _block_operations_2016(x, w,
                               b=None,
                               s=1,
                               pad='SAME',
                               batch_norm=True,
                               is_training=True,
                               activation=None):
        if batch_norm:
            x = Resnet._batch_norm_layer(x, is_training=is_training)
        else:
            x = tf.nn.bias_add(x, b)

        if activation:
            x = activation(x)

        x = tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding=pad)

        return x

    @staticmethod
    def _batch_norm_layer(x,is_training=True):

        bn = batch_norm(x, decay=0.999, center=True, scale=True,
                        updates_collections=None,
                        is_training=is_training,
                        reuse=None,
                        trainable=True)
#        else:
#            bn = batch_norm(x, decay=0.999, center=True, scale=True,
#                            updates_collections=None,
#                            is_training=False,
#                            reuse=True,
#                            trainable=True)
        return bn


    @staticmethod
    def _make_weights(shp, name):

        std = None

        # FC layer
        if len(shp) == 2:
            std = math.sqrt(1.0/shp[0])
        # Conv layer
        elif len(shp) == 4:
            k_sqr_d = float(shp[0] * shp[1] * shp[2])
            std = math.sqrt(2.0 / k_sqr_d)
            None
        else:
            raise Exception('Shape should be an array of length 2 or 4')

        return tf.Variable(tf.truncated_normal(shp, stddev=std), name=name)

    @staticmethod
    def _max_pool(img, k, pad='VALID'):
        ks = [1, k, k, 1]
        return tf.nn.max_pool(img, ksize=ks, strides=ks, padding=pad)

    @staticmethod
    def _avg_pool(x, k):
        ks = [1, k, k, 1]
        return tf.nn.avg_pool(x, ksize=ks, strides=ks, padding='VALID')

    @staticmethod
    def _fc(x, w, b, act):
        if act:
            return act(tf.add(tf.matmul(x, w), b))
        else:
            return tf.add(tf.matmul(x, w), b)

class DenseNet:
    @staticmethod
    def get_network(x, num_blocks, layers_per_block=5, growth_rate=12, is_training=True):
        n_classes = 5

        shp =x.get_shape().as_list()
        batch_size = shp[0]
        channels = shp[3]

        in_dim = 16

        with tf.variable_scope('conv1'):
            weights = DenseNet._make_weights([3,3,channels,in_dim], 'weights')
            x = DenseNet._block_operations(x,
                                           weights,
                                           s=2,
                                           pad='VALID',
                                           activation=tf.nn.relu)



        for i in range(1, num_blocks+1):
            # Dense Bloack
            scope = 'block_{}'.format(i)
            with tf.variable_scope(scope):
                w, in_dim = DenseNet._make_block_weights(layers_per_block,in_dim,growth_rate,scope)
                x = DenseNet._dense_block(x, w, layers_per_block, is_training=is_training)

            scope = 'transition' + scope
            # Transition Block
            if i != num_blocks:
                with tf.variable_scope():
                    w = DenseNet._make_weights([1,1,in_dim,in_dim], scope)
                    x = DenseNet._transition_layer(x, w)

        # the paper calls for global average pooling to do that we need to
        # convolve so that we end up with a channel for each class
        with tf.variable_scope('out'):
            w = DenseNet._make_weights([1,1,in_dim,n_classes], 'weights')
            x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID')

            shp = x.get_shape().as_list()

            x = DenseNet._avg_pool(x, shp[1])

        # this will get softmaxed in the cost function
        return x



    @staticmethod
    def _dense_block(x, w, num_layers, is_training=True):
        for i in range(num_layers):
            x_out = DenseNet._block_operations(x, w[i], is_training=is_training)
            x = tf.concat(3, [x, x_out])

        return x

    @staticmethod
    def _transition_layer(x, w):
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')
        x = DenseNet._avg_pool(x, 2)



    @staticmethod
    def _block_operations(x, w,
                          b=None,
                          s=3,
                          pad='SAME',
                          batch_norm=True,
                          is_training=True,
                          activation=None):

        if batch_norm:
            x = DenseNet._batch_norm_layer(x, is_training=is_training)
        else:
            x = tf.nn.bias_add(x, b)

        if activation:
            x = activation(x)

        x = tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding=pad)

        return x


    @staticmethod
    def _batch_norm_layer(x,is_training=True):
        bn = None

        if is_training:
            bn = batch_norm(x, decay=0.999, center=True, scale=True,
                            updates_collections=None,
                            is_training=True,
                            reuse=None, # is this right?
                            trainable=True)#,
                            #scope=scope_bn)
        else:
            bn = batch_norm(x, decay=0.999, center=True, scale=True,
                            updates_collections=None,
                            is_training=False,
                            reuse=True, # is this right?
                            trainable=True)#,
                            #scope=scope_bn)
        return bn


    @staticmethod
    def _avg_pool(x, k):
        ks = [1, k, k, 1]
        return tf.nn.avg_pool(x, ksize=ks, strides=ks, padding='VALID')

    @staticmethod
    def _make_weights(shp, name):

        std = None

        # FC layer
        if len(shp) == 2:
            std = math.sqrt(1.0/shp[0])
        # Conv layer
        elif len(shp) == 4:
            k_sqr_d = float(shp[0] * shp[1] * shp[2])
            std = math.sqrt(2.0 / k_sqr_d)
            None
        else:
            raise Exception('Shape should be an array of length 2 or 4')

        return tf.Variable(tf.truncated_normal(shp, stddev=std))

    @staticmethod
    def _make_block_weights(num_layers, in_dim, growth_rate, scope):
        weights = []

        for i in range(num_layers):
            weights.append(DenseNet._make_weights([3,3,in_dim,growth_rate], '{}weights{}'.format(scope,i)))
            in_dim += growth_rate


        return weights, in_dim









