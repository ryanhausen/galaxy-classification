import os
from datahelper import DataHelper
from network import CandleNet, ExperimentalNet, Resnet
#from resnet import res_net
from math import sqrt
import time
import colorama 


import tensorflow as tf

colorama.init(autoreset=True)

batch_size = 5
train_size = 0.8
display_step = 100
n_classes = 5
# used to be .0001
learning_rate = .0001
momentum = 0.9
model_dir = '../models/'
train_progress = '../report/train_progress.csv'
test_progress = '../report/test_progress.csv'

# make sure we have a place to save the progress 
for new_dir in ['models', 'report']:
    if new_dir not in os.listdir('../'):
        os.mkdir('../' + new_dir)

train = True


#input
x = tf.placeholder(tf.float32, [batch_size,84,84,4])
y = tf.placeholder(tf.float32, [None, n_classes])

#net = ExperimentalNet.get_network(x)
#net = res_net(x)
net = Resnet.get_network(x)

cost = tf.reduce_mean(tf.squared_difference(net, y))

optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)

rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(net,y)))

init = tf.initialize_all_variables()

saver = tf.train.Saver()

# train, test, and save model
with tf.Session() as sess:
    sess.run(init)

    if train:
        # train
        epoch = 1
        while True:  # epoch <= epochs:
            epoch_start = time.time()
            print 'Training Epoch {}...'.format(epoch)
            # get data, test_idx = 19000 is ~83% train test split
            dh = DataHelper(batch_size=batch_size,
                            train_size=train_size, 
                            shuffle_train=False)

            step = 1
            while dh.training:
                batch_xs, batch_ys = dh.get_next_batch()

                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

                if step % display_step == 0:
                    acc = sess.run(rmse, feed_dict={x: batch_xs, y: batch_ys})
                    loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})

                    print "Iter " + str(step * batch_size) + \
                          ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                          ", Training RMSE= " + "{:.5f}".format(acc)

                    with open(train_progress, mode='a') as f:
                        f.write('{},{},{},{}\n'.format(epoch,
                                                       (step * batch_size),
                                                       acc,
                                                       loss))

                step += 1

            print 'Saving checkpoint'
            saver.save(sess, model_dir, global_step=epoch)
            print 'Epoch {} finished'.format(epoch)

            print 'Testing...'
            # test
            test_step = 1
            test_rmse = 0.0
            test_size = 0
            while dh.testing:
#                start = (test_step - 1) * batch_size
#                end = test_step * batch_size
                test_size += batch_size
                batch_xs, batch_ys= dh.get_next_batch()
                 

                _rmse = sess.run(rmse, feed_dict={x: batch_xs, y: batch_ys})
                _rmse = pow(_rmse, 2) * batch_size
                test_rmse += _rmse

                test_step += 1

            test_rmse = sqrt(test_rmse / float(test_size))
            print colorama.Fore.YELLOW + 'Test RMSE:{}'.format(test_rmse)

            with open(test_progress , mode='a') as f:
                f.write('{},{}\n'.format(epoch, test_rmse))

            print 'Time for epoch {}'.format(time.time() - epoch_start)

            epoch += 1
    else:
        ckpt = tf.train.get_checkpoint_state(model_dir)

        dh = DataHelper(batch_size, test_size)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print 'no checkpoint found...'

        batch_xs, _ = dh.get_next_batch()

        predictions = sess.run(net, feed_dict={x: batch_xs})

        print predictions
