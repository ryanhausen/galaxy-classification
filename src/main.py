import sys
import json
import time

from network import Resnet
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

        channels = len(params['bands'])

        x = tf.placeholder(tf.float32, [None,84,84,channels])
        y = tf.placeholder(tf.float32, [None, params['n_classes']])

        net = Resnet.get_network(x,
                                 params['block_config'],
                                 params['train'],
                                 params['global_avg_pool'],
                                 params['2016_update'])
                                 
        if params['train']:
            _train_network(net)
        else:
            _use_network(net)

    except KeyboardInterrupt:
        print '\nInterrupted'    
        run = 'y' == raw_input('Run eval statistics?(y/[n])')
        if run:
            print 'Not implemented yet'
    
            
def _train_network(net):
    global params
    global x
    global y

    iters = tf.Variable(0, trainable=False)
    learning_rate = None
    if params['decay_steps']:
        learning_rate = tf.train.exponential_decay(params['start_learning_rate'],
                                                   iters,
                                                   params['decay_steps'],
                                                   params['decay_base'])
    else:
        learning_rate = tf.Variable(params['start_learning_rate'])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, y))
    
    # find a way tp paramertize the optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           params['momentum'],
                                           params['nesterov'])
    optimize = optimizer.minimize(cost, global_step=iters)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    learning_rate_reduce = params['learning_rate_reduce']

    start = time.time()
    # this should have a more general implementation, we chose 0 because 
    # accuracy will grow as it improves
    top_result = 0.0
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(1, params['epoch_limit']+1):
            if params['print']:
                print epoch
            
            dh = DataHelper(batch_size=params['batch_size'],
                        train_size=params['train_size'],
                        label_noise=params['label_noise'],
                        bands=params['bands'],
                        transform_func=params['trans_func'])

            if learning_rate_reduce and epoch in learning_rate_reduce:
                sess.run(learning_rate.assign(learning_rate.eval() / 10.0))

            while dh.training:
                batch_xs, batch_ys = dh.get_next_batch()
                sess.run(optimize, feed_dict={x:batch_xs, y:batch_ys})

                if iters.eval() % 20 == 0:
                    evaluate.evaluate(sess, net, x, y, batch_xs, batch_ys, None)

            #testing
            batch_xs, batch_ys = dh.get_next_batch()
            results = evaluate.evaluate(sess, net, x, y, batch_xs, batch_ys, None, rtrn=params['rtrn_eval'])

             
            if params['save_progress'] and results[0] > top_result:
                if params['print']:
                    print 'Saving checkpoint'
                
                saver.save(sess, params['model_dir'], global_step=iters)
                top_result = results[0]



            #   cPickle.dump(lbls, open('../report/model_out/lbls_{}.p'.format(epoch), 'wb'))
            #   cPickle.dump(preds, open('../report/model_out/preds_{}.p'.format(epoch), 'wb'))
    if params['print']:
        print 'Epoch took {} seconds'.format(complete = time.time() - start)
    
    
    if params['rtrn_eval']:
        print top_result


def _use_network(net):
    raise NotImplementedError()

def _type_convert(dictionary):
    for k in dictionary.keys():
        if dictionary[k] == 'true':
            dictionary[k] = True
        elif dictionary[k] == 'false':
            dictionary[k] = False
    
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





















































