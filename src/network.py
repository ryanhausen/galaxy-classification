import math
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


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
            return act(tf.add(tf.matmul(x, w), b))

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
            output = fc(fc2, _weights['out'], _biases['out'], tf.nn.softmax)
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

            #print 'conv1: {}'.format(conv1.get_shape())

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

        # Model Helpers --------------------------------------------------------

        c1 = conv_net(c1)
        c2 = conv_net(c2)
        c3 = conv_net(c3)
        c4 = conv_net(c4)
        
        # feed this into one fully connected layer
        #cmb = tf.pack([c1,c2,c3,c4], axis=1)
        cmb = tf.concat(1, [c1,c2,c3,c4]) 
        
        # fully connected
        fc1 = fc(cmb, fc_weights['wf1'], fc_biases['bf1'], tf.nn.relu)
        fc2 = fc(fc1, fc_weights['wf2'], fc_biases['bf2'], tf.nn.relu)
        
        # output
        output = fc(fc2, fc_weights['out'], fc_biases['out'], tf.nn.softmax)
        
        return output
     
class Resnet:
    @staticmethod
    def get_network(x, num_blocks, is_training=True):
        n_classes = 5        
        
        shp =x.get_shape().as_list()
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

        #block operations        
        for i in range(1, num_blocks+1):
            
            with tf.variable_scope('block{}'.format(i)):
                weights = []            
                first = i == 1
                increase_dim = i % 3 == 0 or first
                
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
                        
                x = Resnet._building_block(x, 
                                           weights, 
                                           increase_dim=increase_dim, 
                                           first=first)
                                           
                if increase_dim:
                    in_dim *= 2
                if first:
                    in_dim *= 2
                
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
        # removed activation function for tf crossentropy loss
        output = Resnet._fc(fc2, fc_weights['out'], fc_biases['out'], None)
                
        return output
        
    @staticmethod
    def _building_block(x, ws, activation=tf.nn.relu, increase_dim=False, first=False):
        dim_stride = 1        
        pad = 'SAME'

        if increase_dim and not first:
            dim_stride = 2
            pad = 'VALID'
        
        # first 1x1 conv
        f_x = Resnet._block_operations(x, ws[0], s=1, pad='VALID', activation=activation)
        
        # 3x3 conv
        f_x = Resnet._block_operations(f_x, ws[1], s=dim_stride, pad=pad, activation=activation)
        
        # second 1x1 conv
        f_x = Resnet._block_operations(f_x, ws[2], s=1, pad='VALID', activation=None)
                
        if increase_dim:
            x = Resnet._block_operations(x, ws[3], s=dim_stride, pad='VALID', activation=None)
            
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


    #https://github.com/tensorflow/models/blob/master/inception/inception/slim/ops.py#L116
    def _batch_norm(img, activation=None, decay=0.999, center=None, scale=None):
        
        img_shape = inpt.get_shape()
        img_axis = list(range(len(inputs_shape) - 1))
        params_shape = inputs_shape[-1:]
        
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta = variables.variable('beta',
                                    params_shape,
                                    initializer=tf.zeros_initializer,
                                    trainable=trainable,
                                    restore=restore)
        if scale:
            gamma = variables.variable('gamma',
                                     params_shape,
                                     initializer=tf.ones_initializer,
                                     trainable=trainable,
                                     restore=restore)        
            
        if is_training:
            mean, variance = tf.nn.moments(img, axis)
        else:
            None
            
    
    @staticmethod
    def _max_pool(img, k, pad='VALID'):
        ks = [1, k, k, 1]
        return tf.nn.max_pool(img, ksize=ks, strides=ks, padding=pad)
            
    @staticmethod
    def _fc(x, w, b, act):
        if act:
            return act(tf.add(tf.matmul(x, w), b))
        else:
            return tf.add(tf.matmul(x, w), b)
        
    @staticmethod
    def _building_block_bottlenext(x, w, b):
        
        conv1 = Resnet._conv2d(x, w[0], b[0])
        # batch_norm
        #norm1 =         
        
        conv2 = Resnet._conv2d(conv1, w[1], b[1])
        conv3 = Resnet._conv2d(conv2, w[2], b[2])        
        
        return tf.nn.relu(conv3 + x)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        