
import datahelper_mapper as dh
import ResNetMapper as network
from resnet import print_total_params
import evaluate


import tensorflow as tf

x = tf.placeholder(tf.float32, [None,40,40,1])
y = tf.placeholder(tf.float32, [None,40,40,6])
iters = tf.Variable(1, trainable=False)
iter_limit = 10000
batch_size = 50

learning_rate = tf.train.exponential_decay(0.001,
                                           iters,
                                           5000,
                                           0.96)

network.ResNetMapper._DATA_FORMAT = 'NHWC'
net = network.ResNetMapper.build_graph(x, [3,3,3,3], True, None)

with tf.variable_scope('loss'):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=net)
    loss = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    grads = optimizer.compute_gradients(loss)
    with tf.name_scope('clipping'):
        grads = [(tf.clip_by_value(grad, -1.5, 1.5), var) for grad,var in grads]
    update = optimizer.apply_gradients(grads, global_step=iters)

top_result = 0.0

data = dh.DataHelper()

with tf.name_scope('metrics'):
    evaluate.evaluate_tensorboard(net, y)
summaries = tf.summary.merge_all()

init = tf.global_variables_initializer()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print_total_params()

with tf.Session(config=config) as sess:
    sess.run(init)

    trainWriter = tf.summary.FileWriter("../report/tf-log/train", graph=sess.graph)
    testWriter = tf.summary.FileWriter("../report/tf-log/test", graph=sess.graph)

    current_iter = iters.eval()
    while current_iter < iter_limit:
        print('iter ', current_iter)
        tf.logging.info('Training iter: {}'.format(current_iter))

        xs, ys = data.next_batch(batch_size, True, False)
        batch = {x:xs, y:ys}

        if current_iter%10==0:
            tf.logging.info('Evaluating')
            s = sess.run(summaries, feed_dict=batch)
            trainWriter.add_summary(s, current_iter)

        if current_iter%100==0:
            tf.logging.info('Testing')
            test_xs, test_ys = data.next_batch(-1, False, False)
            test_data = {x: test_xs, y:test_ys}
            s = sess.run(summaries, feed_dict=test_data)
            testWriter.add_summary(s, current_iter)
