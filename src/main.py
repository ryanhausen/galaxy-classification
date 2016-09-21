import os
from datahelper import DataHelper
from network import CandleNet, ExperimentalNet, Resnet
from math import sqrt
import numpy as np
import time
import colorama 
import cPickle

import tensorflow as tf

colorama.init(autoreset=True)

# PARAMS
batch_size = 110
train_size = 0.8
n_classes = 5
decay_steps = 150
decay_base = 0.96
epoch_reduce = None#[100, 120]
# used to be .0001
start_learning_rate = .1
momentum = 0.9
bands_to_use = ['v','z','h','j']
block_config = [3,8,36,3]
trans_func = None#lambda x: np.log10(x + 1.0)
band_trans_func = None#lambda x: (((1-.0001)*(x-np.min(x)))/(np.max(x)-np.min(x))) + .0001
label_noise_stddev = 0.001

display_step = 10
model_dir = '../models/'
train_progress = '../report/train_progress.csv'
test_progress = '../report/test_progress.csv'
save_progress = True
train = True



# make sure we have a place to save the progress 
for new_dir in ['models', 'report']:
    if new_dir not in os.listdir('../'):
        os.mkdir('../' + new_dir)



#input
x = tf.placeholder(tf.float32, [batch_size,84,84,len(bands_to_use)])
y = tf.placeholder(tf.float32, [batch_size, n_classes])

global_step = tf.Variable(0, trainable=False)
#learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_base)
learning_rate = tf.Variable(start_learning_rate)

#net = ExperimentalNet.get_network(x)
#net = res_net(x)
net = Resnet.get_network(x, block_config,
                         global_avg_pool=True,
                         use_2016_update=True, 
                         is_training=train)

#cost = tf.reduce_mean(tf.squared_difference(net, y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, y))

optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True).minimize(cost, global_step=global_step)

rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(tf.nn.softmax(net),y)))

rmse_acc = tf.squared_difference(tf.nn.softmax(net), y)
#rmse_fin = tf.sqrt(tf.reduce_mean(val))

init = tf.initialize_all_variables()

saver = tf.train.Saver()

#raise Exception('Network size test')
# train, test, and save model
with tf.Session() as sess:
    sess.run(init)

    min_rmse = 1.0
    if train:
        # train
        epoch = 1
        while True:  # epoch <= epochs:
            print colorama.Fore.BLUE + 'Current Learning Rate {}, Global Step:{}'.format(learning_rate.eval(), global_step.eval())
            epoch_start = time.time()
            print 'Training Epoch {}...'.format(epoch)
            dh = DataHelper(batch_size=batch_size,
                            train_size=train_size, 
                            shuffle_train=True,
                            label_noise=label_noise_stddev,
                            bands=bands_to_use,
                            transform_func=trans_func,
                            band_transform_func=band_trans_func)
                            
            if epoch_reduce and epoch in epoch_reduce:
                sess.run(learning_rate.assign(learning_rate.eval() / 10.0))                            
                            
            step = 1
            while dh.training:
                batch_xs, batch_ys = dh.get_next_batch()

                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

                if step % display_step == 0:
                    acc = sess.run(rmse, feed_dict={x: batch_xs, y: batch_ys})
                    loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})

                    print "Iter " + str(step * batch_size) + \
                          ", Minibatch Loss= " + "{}".format(loss) + \
                          ", Training RMSE= " + "{}".format(acc)

                    with open(train_progress, mode='a') as f:
                        f.write('{},{},{},{}\n'.format(epoch,
                                                       (step * batch_size),
                                                       acc,
                                                       loss))

                step += 1
                

            
            print 'Epoch {} finished'.format(epoch)

            print 'Testing...'
            
            lbls = []
            preds = []
            sources = []
            
            # test
            test_step = 1
            test_rmse = 0.0
            test_size = 0
            while dh.testing:
                test_size += batch_size
                ids, batch_xs, batch_ys= dh.get_next_batch(include_ids=True)

                sources.append(ids)
                lbls.append(batch_ys)
                preds.append(sess.run(tf.nn.softmax(net), feed_dict={x: batch_xs}))


                _rmse = sess.run(rmse_acc, feed_dict={x: batch_xs, y: batch_ys})
                test_rmse += _rmse.mean(axis=1).sum()

                test_step += 1

            test_rmse = sqrt(test_rmse / float(test_size))
            print colorama.Fore.YELLOW + 'Test RMSE:{}'.format(test_rmse)

            if save_progress and test_rmse < min_rmse:
                print 'Saving checkpoint'
                saver.save(sess, model_dir, global_step=epoch)
                min_rmse = test_rmse
                
                cPickle.dump(lbls, open('../report/model_out/lbls_{}.p'.format(epoch), 'wb'))
                cPickle.dump(preds, open('../report/model_out/preds_{}.p'.format(epoch), 'wb'))
                cPickle.dump(sources, open('../report/model_out/srcs_{}.p'.format(epoch), 'wb'))


            with open(test_progress , mode='a') as f:
                f.write('{},{}\n'.format(epoch, test_rmse))

            print 'Time for epoch {} seconds'.format(time.time() - epoch_start)

            epoch += 1
    else:
        img_id = 'deep2_4222'     
        
        ckpt = tf.train.get_checkpoint_state(model_dir)

        dh = DataHelper(batch_size=None)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print 'no checkpoint found...'


        output = ''
        while dh.testing:
            print dh._idx
            
            x_in, y = dh.get_next_example(img_id=img_id)
            y_hat = sess.run(net, feed_dict={x: x_in})
            
            output += ','.join(['{0:.9f}'.format(i) for i in np.append(y,y_hat)]) + '\n'

        with open('../report/prediction_output.csv', 'a') as f:
            f.write(output)