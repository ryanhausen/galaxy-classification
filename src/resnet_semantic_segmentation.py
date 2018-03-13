import types

import tensorflow as tf
from tf_src import resnet
from tf_src import tf_logger as log

var_init = tf.variance_scaling_initializer
conv2d = tf.layers.conv2d

class Model:
    DATA_FORMAT = 'channels_first'
    INIT_FILTERS = 4
    BLOCK_CONFIG = [2, 4, 4, 8]

    def __init__(self, dataset, is_training):
        self.dataset = dataset
        self.is_training = is_training
        self.num_classes = dataset.NUM_LABELS

        self._graph = None
        self._train = None
        self._optimizer = None
        self._test = None
        self._inference = None

        self._train_metrics = None
        self._test_metrics = None

    def graph(self, x):
        if self._graph:
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
                           kernel_initializer=var_init(),
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

                    for b_idx in range(-1, -(Model.BLOCK_CONFIG[s_idx]+1), -1):
                        with tf.variable_scope('block{}'.format(b_idx)):
                            log_shape(s_idx, b_idx, x)

                            if b_idx==-1 and s_idx>-4:
                                resample_op = Model.up_sample(x)
                                projection_op = Model.up_project(x)
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
                           self.num_classes,
                           1,
                           kernel_initializer=var_init(),
                           padding='same',
                           data_format=Model.DATA_FORMAT)
                log.debug('[out_conv]::{}'.format(x.shape.as_list()))

            return x

        self._graph = model_fn
        return self._graph(x)

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

    @staticmethod
    def up_sample(x):
        def _f(_x):
            w, h = _x.shape.as_list()[1:3]
            _x = tf.image.resize_images(_x,
                                        (w*2, h*2),
                                        method=tf.image.ResizeMethod.BICUBIC)
            return _x


        if Model.DATA_FORMAT=='channels_first':
            def f(_x):
                _x = tf.transpose(_x, [0, 2, 3, 1])
                _x = _f(_x)
                _x = tf.transpose(_x, [0, 3, 1, 2])

                return _x
        else:
            def f(_x):
                return _f(_x)

        return f

    @staticmethod
    def up_project(x):
        return Model.up_sample(x)

    def train(self):
        x, y = self.dataset.train
        logits = self.graph(x)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                            labels=y)
        optimize = self.optimizer(loss)

        metrics = self.train_metrics(logits, y)

        return optimize, metrics

    def test(self):
        x, y = self.dataset.test

        logits = self.graph(x)

        metrics = self.test_metrics(logits, y)

        return logits, metrics

    def train_metrics(self, logits, y):
        if self._train_metrics:
            return self._train_metrics(logits, y)

        def f(logits, ys):
            log.warn('No training metrics set')
            return logits, ys

        self._train_metrics = f

        return self._train_metrics(logits, y)

    def test_metrics(self, logits, y):
        if self._test_metrics:
            return self._test_metrics(logits, y)

        def f(logits, y):
            log.warn('No testing metrics set')
            return tf.constant(0)

        self._test_metrics = f
        return self._test_metrics(logits, y)

    def optimizer(self, loss):
        if self._optimizer:
            return self._optimizer(loss)

        def optimize(loss):
            log.warn('No optimizer set')

        self._optimizer = optimize

        return self._optimizer(loss)

    def inference(self, x):
        return tf.nn.softmax(self.graph(x))


def main():
    info = f"""
    CANDELS Morphological Classification -- Semantic Segmentation
    ResNet
    BLOCK_CONFIG:   {Model.BLOCK_CONFIG}
    INIT_FILTERS:   {Model.INIT_FILTERS}
    DATA_FORMAT:    {Model.DATA_FORMAT}
    """

    tf.logging.set_verbosity(tf.logging.DEBUG)
    print(info)

    in_shape = [5,1,40,40]
    expect_out_shape = [5,5,40,40]

    x = tf.placeholder(tf.float32, shape=in_shape)

    mock_dataset = types.SimpleNamespace(NUM_LABELS=5)
    Model.DATA_FORMAT = 'channels_first'
    m = Model(mock_dataset, True)

    out = m.graph(x)

    assert out.shape.as_list()==expect_out_shape, "Incorrect Shape"

if __name__=='__main__':
    main()