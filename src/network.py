import tensorflow as tf


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
            output = fc(fc2, _weights['out'], _biases['out'], tf.sigmoid)
            return output

        # Model Helpers --------------------------------------------------------


        # Model weights and biases
        weights = {
            # 6x6 conv, 3-channel input, 32-channel outputs
            'wc1': tf.Variable(tf.truncated_normal([6, 6, channels, 32], stddev=0.01)),
            # 5x5 conv, 32-channel inputs, 64-channel outputs
            'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01)),
            # 3x3 conv, 64-channel inputs, 128-channel outputs
            'wc3': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.01)),
            # 3x3 conv, 128-channel inputs, 128-channel outputs
            'wc4': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)),
            # fully connected, 512 inputs, 2048 outputs
            'wf1': tf.Variable(tf.truncated_normal([4608, 2048], stddev=0.001)),
            # fully coneected 2048 inputs, 2048 outputs
            'wf2': tf.Variable(tf.truncated_normal([2048, 2048], stddev=0.001)),
            # 2048 inputs, 5 outputs (class prediction)
            'out': tf.Variable(tf.truncated_normal([2048, n_classes], stddev=0.01))
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
