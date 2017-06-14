import sys
import json
import time
import string

import _pickle as cPickle


from network import Resnet, ResNet
from datahelper import DataHelper
import evaluate

import tensorflow as tf

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

        x = tf.placeholder(tf.float32, [None,84,84,channels])
        y = tf.placeholder(tf.float32, [None, params['n_classes']])

#        net = Resnet.get_network(x,
#                                 params['block_config'],
#                                 params['train'],
#                                 params['global_avg_pool'],
#                                 params['2016_update'])

        net = ResNet.build_graph(x, params['block_config'], params['train'])
        eval_net = ResNet.build_graph(x, params['block_config'], False)

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
        learning_rate = tf.Variable(params['start_learning_rate'])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))

    # find a way tp paramertize the optimizer
#    optimizer = tf.train.MomentumOptimizer(learning_rate,
#                                           params['momentum'],
#                                           params['nesterov'])
    optimizer = tf.train.AdamOptimizer(learning_rate)
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    optimize = optimizer.minimize(cost, global_step=iters)

#    eval_ops = [evaluate.top_1(eval_net, y),
#                evaluate.top_2(eval_net, y),
#                evaluate.cross_entropy(eval_net, y),
#                evaluate.class_accuracy_part1(eval_net, y)]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    learning_rate_reduce = params['learning_rate_reduce']

    # this should have a more general implementation, we chose 0 because
    # accuracy will grow as it improves
    top_result = 0.0
    dh = DataHelper(batch_size=params['batch_size'],
                    train_size=params['train_size'],
                    label_noise=params['label_noise'],
                    bands=params['bands'],
                    transform_func=eval(params['trans_func']) if params['trans_func'] else None)

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(1, params['epoch_limit']+1):
            start = time.time()

            if params['print']:
                tf.logging.info(f'Epoch:{epoch}')

            if learning_rate_reduce and epoch in learning_rate_reduce:
                sess.run(learning_rate.assign(learning_rate.eval() / 10))

            if params['print']:
                tf.logging.info('Training...')

            file_writer = tf.summary.FileWriter('./tf-log', sess.graph)

            total = len(dh._train_imgs)
            while dh.training:
                batch_xs, batch_ys = dh.get_next_batch()
                sess.run(optimize, feed_dict={x:batch_xs, y:batch_ys})

                if params['print']:
                    print(f'{round(100 * (dh._idx/total), 2)}% complete', end='\r')

                if iters.eval() % 10 == 0:
                    if params['print']:
                        tf.logging.info('Evaluating...')
                        evaluate.evaluate(sess, eval_net, x, y, batch_xs, batch_ys, params['train_progress'])
#                        evals = sess.run(eval_ops, feed_dict={x:batch_xs, y:batch_ys})
#                        evals[-1] = evaluate.class_accuracy_part2(evals[-1])
#                        evaluate.save(evals, params[params['train_progress']])

                    if params['print']:
                        tf.logging.info('Training...')

            if params['print']:
                tf.logging.info('Testing...')

            srcs, batch_xs, batch_ys = dh.get_next_batch(include_ids=True)
            evals = evaluate.evaluate(sess, eval_net, x, y, batch_xs, batch_ys, params['test_progress'])
#            evals = sess.run(eval_ops, feed_dict={x:batch_xs, y:batch_ys})
#            evals[-1] = evaluate.class_accuracy_part2(evals[-1])
#            evaluate.save(evals, params[params['test_progress']])

            if params['save_progress'] and evals[0] > top_result:
                if params['print']:
                    tf.logging.info('Saving checkpoint')
                saver.save(sess, params['model_dir'], global_step=iters)
                top_result = evals[0]

                cPickle.dump(srcs, open('../report/model_out/srcs_{}.p'.format(epoch), 'wb'))
                cPickle.dump(batch_ys, open('../report/model_out/lbls_{}.p'.format(epoch), 'wb'))
                cPickle.dump(sess.run(eval_net, feed_dict={x: batch_xs}), open('../report/model_out/preds_{}.p'.format(epoch), 'wb'))

            if params['print']:
                tf.logging.info(f'Epoch {epoch} took {(time.time() - start)} seconds')

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





















































