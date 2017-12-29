import sys
import json
import time
import string

import _pickle as cPickle


from network import ResNet
#from resnet import ResNet
import resnet
from datahelper import DataHelper
import evaluate

import tensorflow as tf
from tensorflow.python.client import timeline


params = None
x = None
y = None
logger = None

def main(config=None):
    try:
        global params
        global x
        global y

        # config should be a dictionary
        if config:
            params = _type_convert(config)
        else:
            # load params
            with open('params.json') as f:
                params = _type_convert(json.load(f))

        # add some code to validate dictionary
        if params['print']:
            tf.logging.set_verbosity(tf.logging.INFO)

        channels = len(params['bands'])

        x = tf.placeholder(tf.float32, [None,60,60,channels])
        y = tf.placeholder(tf.float32, [None, params['n_classes']])

        net = ResNet.build_graph(x, params['block_config'], params['train'])
        eval_net = net# ResNet.build_graph(x, params['block_config'], False)
        resnet.print_total_params()

        if params['train']:
            _train_network(net, eval_net)
        else:
            _use_network(eval_net)

    except KeyboardInterrupt:
        print('\nInterrupted')
        run = 'y' == input('Run eval statistics?(y/[n])')
        if run:
            print('Not implemented yet')


def _train_network(net, eval_net):
    global params
    global x
    global y

    iters = tf.Variable(1, trainable=False)
    learning_rate = None
    if params['decay_steps']:
        learning_rate = tf.train.exponential_decay(params['start_learning_rate'],
                                                   iters,
                                                   params['decay_steps'],
                                                   params['decay_base'])
    else:
        learning_rate = tf.Variable(params['start_learning_rate'], trainable=False)
    with tf.name_scope('loss'):
        #loss_weights =  1.003 - tf.reduce_max(y, axis=1)

        loss = tf.losses.softmax_cross_entropy(y,
                                               net,
                                               weights=1.0,
                                               reduction=tf.losses.Reduction.MEAN)
        #loss = tf.nn.softmax_cross_entropy_with_logits(logits=net,
        #                                               labels=y,
        #                                               weights=loss_weights,
        #                                               reduction=tf.losses.Reduction.MEAN)
        #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=params['momentum'])
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        grads = optimizer.compute_gradients(loss)
        with tf.name_scope('clipping'):
            grads = [(tf.clip_by_value(grad, -1.5, 1.5), var) for grad, var in grads]
        update = optimizer.apply_gradients(grads, global_step=iters)

    # with tf.name_scope('grads'):
    #     for grad, var in grads:
    #         tf.summary.histogram(f"{var.name.split(':')[0]}", grad)

    # with tf.name_scope('weights'):
    #     for grad, var in grads:
    #         tf.summary.histogram(f"{var.name.split(':')[0]}", var)

    learning_rate_reduce = params['learning_rate_reduce']

    # this should have a more general implementation, we chose 0 because
    # accuracy will grow as it improves
    top_result = 0.0
    dh = DataHelper(batch_size=params['batch_size'],
                    train_size=params['train_size'],
                    label_noise=params['label_noise'],
                    bands=params['bands'],
                    transform_func=eval(params['trans_func']) if params['trans_func'] else None)

    with tf.name_scope('metrics'):
        evaluate.evaluate_tensorboard(eval_net, y)
    summaries = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)

        trainWriter = tf.summary.FileWriter('../report/tf-log/train', graph=sess.graph)
        testWriter = tf.summary.FileWriter('../report/tf-log/test', graph=sess.graph)
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        run_options = None
        run_metadata = None

        top_result = 0
        while iters.eval() < params['iter_limit']:
            current_iter = iters.eval()

            if learning_rate_reduce and current_iter in learning_rate_reduce:
                sess.run(learning_rate.assign(learning_rate.eval() / 10))

            if params['print']:
                tf.logging.info(f"Training iter:{current_iter}")

            batch_xs, batch_ys = dh.get_next_batch(iter_based=True)
            batch = {x:batch_xs, y:batch_ys}
            sess.run(update, feed_dict=batch)

            if current_iter%10==0:
                if params['print']:
                    tf.logging.info("Evaluating")
                s = sess.run(summaries, feed_dict=batch)
                trainWriter.add_summary(s, current_iter)

            if current_iter%100==0:
                if params['print']:
                    tf.logging.info('Testing')

                batch_xs, batch_ys = dh.get_next_batch(force_test=True)
                batch[x]=batch_xs; batch[y]=batch_ys
                s = sess.run(summaries, feed_dict=batch)
                testWriter.add_summary(s, current_iter)

                evals = evaluate.evaluate(sess,
                                          eval_net,
                                          x, y,
                                          batch_xs, batch_ys,
                                          params['test_progress'])

                if params['save_progress'] and evals[0] > top_result:
                    if params['print']:
                        tf.logging.info('Saving checkpoint')
                    saver.save(sess, params['model_dir'], global_step=iters)
                    top_result = evals[0]


    # This needs to be printed so that the async trainer can see the result
    if params['rtrn_eval']:
        print(top_result)

def _use_network(net):
    raise NotImplementedError()

def _type_convert(dictionary):
    for k in dictionary.keys():
        if dictionary[k] == 'true':
            dictionary[k] = True
        elif dictionary[k] == 'false':
            dictionary[k] = False
        elif dictionary[k] == '':
            dictionary[k] = None
        elif type(dictionary[k]) not in  (float, int) and dictionary[k][0] == '[':
            # break up the list
            vals = dictionary[k][1:-1]
            vals = [v.strip() for v in vals.split(',')]

            # figure out what type is in the list
            val  = vals[0]
            is_num = False
            is_float = False

            for v in val:
                if v in string.digits:
                    is_num = True
                elif v == '.':
                    is_float = True

            # this is a list of numns
            if is_num:
                if is_float:
                    vals = [float(v) for v in vals]
                else:
                    vals = [int(v) for v in vals]
            # this is a list of strings
            else:
                vals = [str(v) for v in vals]

            dictionary[k] = vals
        elif type(dictionary[k]) == str:
            is_num = False
            is_float = False
            for s in dictionary[k]:
                if s in string.ascii_letters:
                    is_num = False
                    is_float = False
                    break
                elif s in string.digits:
                    is_num = True
                elif s == '.':
                    is_float = True

            if is_float:
                dictionary[k] = float(dictionary[k])
            elif is_num:
                dictionary[k] = int(dictionary[k])

    return dictionary

if __name__ == '__main__':
    args = None if len(sys.argv) < 2 else dict()

    tmp = None
    for arg in sys.argv[1:]:
        if '-' in arg:
            tmp = arg[1:]
        else:
            args[tmp] = arg
            tmp = None

    main(args)