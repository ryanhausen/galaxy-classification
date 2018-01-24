
import datahelper_mapper as dh
import ResNetMapper as network
from resnet import print_total_params
import evaluate
import tf_logger as log

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


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

log.info('building graph...')
net = network.ResNetMapper.build_graph(x, [3,3,3,3], True, None)
log.info('done')

with tf.variable_scope('loss'):
    loss = evaluate.weighted_cross_entropy(net, y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    grads = optimizer.compute_gradients(loss)
    with tf.name_scope('clipping'):
        grads = [(tf.clip_by_value(grad, -1.5, 1.5), var) for grad,var in grads]
    update = optimizer.apply_gradients(grads, global_step=iters)

top_result = 0.0

data = dh.DataHelper(out_size=40)

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
        log.info('iter {}'.format(current_iter))

        xs, ys = data.next_batch(batch_size, True, False)
        batch = {x:xs, y:ys}
        sess.run(update, feed_dict=batch)

        if current_iter%10==0:
            log.info('Evaluating')
            s = sess.run(summaries, feed_dict=batch)
            trainWriter.add_summary(s, current_iter)

            yhs = sess.run([tf.reshape(tf.nn.softmax(net), [-1,6])], feed_dict={x:xs})
            cm = evaluate.tensorboard_confusion_matrix(yhs, ys)
            testWriter.add_summary(cm, current_iter)


        if current_iter%100==0:
            log.info('Testing')
            test_xs, test_ys = data.next_batch(-1, False, False)
            test_data = {x: test_xs, y:test_ys}
            s = sess.run(summaries, feed_dict=test_data)
            testWriter.add_summary(s, current_iter)

            test_yhs = sess.run([tf.nn.softmax(net)], feed_dict={x:test_xs})
            cm = evaluate.tensorboard_confusion_matrix(test_yhs, test_ys)
            testWriter.add_summary(cm, current_iter)

        current_iter = iters.eval()
