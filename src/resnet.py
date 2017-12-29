import tensorflow as tf
import resnet_model as tf_model

# https://arxiv.org/abs/1512.03385
# https://arxiv.org/abs/1603.05027
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
class ResNet:
    @staticmethod
    def build_graph(x, block_config, is_training, data_format='channels_first'):
        reuse = not is_training
        is_large = sum(block_config) * 2 >= 50
        block_f = tf_model.bottleneck_block if is_large else tf_model.building_block
        num_filters = 32

        if data_format=='channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])

        # initial Convolution
        with tf.variable_scope('initial_conv', reuse=reuse):
            x = tf_model.conv2d_fixed_padding(inputs=x,
                                            filters=num_filters,
                                            kernel_size=5,
                                            strides=2,
                                            data_format=data_format)

        for i in range(len(block_config)):
            with tf.variable_scope(f'block{i}', reuse=reuse):
                strides = 1 if i==0 else 2
                x = tf_model.block_layer(inputs=x,
                                        filters=num_filters,
                                        block_fn=block_f,
                                        blocks=block_config[i],
                                        strides=strides,
                                        is_training=is_training,
                                        name=f'Block-{i}',
                                        data_format=data_format)

            num_filters *= 2

        with tf.variable_scope('out_bn', reuse=reuse):
            x = tf_model.batch_norm_relu(x, is_training, data_format)

        # with tf.variable_scope('global_avg_pooling', reuse=reuse):
        #     x = tf.layers.conv2d(x,
        #                          filters=5,
        #                          kernel_size=[1, 1],
        #                          data_format=data_format)
        #     x = tf.reduce_mean(x, axis=[2, 3])

        with tf.variable_scope('average_pool'):
            x = tf.reduce_mean(x, axis=[2, 3])

        with tf.variable_scope('fc_out'):
            x = tf.layers.dense(x, 5)

        return x

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
    tf.logging.info(f'Memory Estimate:{total_parameters*32/8/1024/1024} MB')