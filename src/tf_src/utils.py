import tensorflow as tf

def configured_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def fetch_iters():
    init = tf.constant_initializer(1, dtype=tf.int32)
    return tf.get_variable('iters',
                           initializer=init,
                           dtype=tf.int32,
                           trainable=False,
                           shape=[])

def print_total_params(print_function=tf.logging.info):
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
    print_function(f'TOTAL NUMBER OF PARAMTERS:{total_parameters}')
    print_function(f'Memory Estimate:{total_parameters*32/8/1024/1024} MB')