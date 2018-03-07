from tf-src import resnet, tf_logger as log
import tensorflow as tf
from tf import variance_scaling_initializer as var_init
from tensorflow.layers import conv2d

class Model:
    DATA_FORMAT = 'channels_first'
    INIT_FILTERS = 4
    BLOCK_CONFIG = [2, 4, 4, 8]
    NUM_CLASSES = 5

    def __init__(self, dataset, is_training):
        self.dataset = dataset
        self.is_training = is_training

    @property
    def graph(self):
        if self._graph is not None:
            return self._graph

        def log_shape(s, b, x):
            shp_str = str(x.shape.as_list())
            log.debug('[segment{}_block{}]::{}'.format(s, b, shp_str))

        concat_axis = 1 if Model.DATA_FORMAT=='channels_first' else 3

        def model_fn(x):
            log.debug('[input]::{}'.format(x.shape.as_list()))

            with tf.variable_scope('in_conv'):
                x = conv2d(x, 
                           Model.INIT_FILTERS,
                           3,
                           weights_initializer=var_init(),
                           padding='same',
                           data_format=Model.DATA_FORMAT)

                log.debug('[in_conv]::{}'.format(x.shape.as_list()))

            # encoder
            encoded = []
            for s_idx in range(len(Model.BLOCK_CONFIG)):
                with tf.variable_scope('segment{}'.format(s_idx)):
                    for b_idx in range(Model.BLOCK_CONFIG[s_idx]):
                        with tf.variable_scope('block{}'.format(b_idx)):
                            log_shape(s_idx, b_idx, x)
                            
                            if b_idx==0 and s_idx>0:
                                resample_op = Model.down_sample(x)
                                projection_op = Model.down_project(x)
                            else:
                                resample_op = None
                                projection_op = None

                            x = resnet.block(x=x,
                                             is_training=self.is_training,
                                             projection_op=projection_op,
                                             resample_op=resample_op,
                                             data_format=Model.DATA_FORMAT)
                encoded.append(x)

            # decoder
            for s_idx in range(-1, -(len(Model.BLOCK_CONFIG)+1), -1):
                with tf.variable_scope('segment{}'.format(s_idx)):
                    x = tf.concat([x, encoded[s_idx]], concat_axis)
                    for b_idx in range(-1, -(block_config[s_idx]+1), -1):
                        log_shape(s_idx, b_idx, x)
                        
                        if b_idx==-1 and s_idx>-4:
                            resample_op = None
                            projection_op = None
                        else:
                            resample_op = None
                            projection_op = None

                        x = resnet.block(x=x,
                                         is_training=self.is_training,
                                         projection_op=projection_op,
                                         resample_op=resample_op,
                                         data_format=Model.DATA_FORMAT)

            with tf.variable_scope('out_conv'):
                x = conv2d(x, 
                           Model.NUM_CLASSES, 
                           1,
                           weights_initializer=var_init(),
                           padding='same',
                           data_format=Model.DATA_FORMAT)
                log.debug('[out_conv]::{}'.format(x.shape.as_list()))

            return x
        
        self._graph = model_fn
        return self._graph

    @staticmethod
    def down_sample(x):
        if Model.DATA_FORMAT=='channels_first':
            ch_idx = 1
            pad_shape = [[0, 0], [0, 0], [1, 1], [1, 1]]
        else:
            ch_idx = 3
            pad_shape = [[0, 0], [1, 1], [1, 1], [0, 0]]

        
        num_filters = x.shape.as_list()[ch_idx] * 2
        def f(_x):
            _x = tf.pad(x, pad_shape)
            _x = resnet.block_conv(x=_x, 
                                   filters=num_filters,
                                   stride=2,
                                   padding='valid',
                                   data_format=Model.DATA_FORMAT)
            return _x
        return f
        
    @staticmethod
    def down_project(x):
        if Model.DATA_FORMAT=='channels_first':
            ch_idx = 1
        else:
            ch_idx = 3

        num_filters = x.shape.as_list()[ch_idx] * 2
        def f(_x):
            _x = resnet.block_conv(x=_x,
                                   filters=num_filters,
                                   stride=2,
                                   kernel_size=1,
                                   padding='valid',
                                   data_format=Model.DATA_FORMAT)
            return _x
        
        return f

    @property
    def train():
        None
    
    @property
    def inference():
        None

    @property
    def test():
        None

