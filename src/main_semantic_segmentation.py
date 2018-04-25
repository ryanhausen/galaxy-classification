import os
from tf_src import tf_logger as log
from galaxies_semantic_segmentation import Dataset
from resnet_semantic_segmentation import Model
from tf_src.utils import configured_session, fetch_iters
import tensorflow as tf

import evaluate

def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)

    params = {
        'data_format':'channels_last',
        'train_dir':'../report/tf-log/train/',
        'test_dir':'../report/tf-log/test/',
        'max_iters':10000,
        'batch_size':75,
        'data_dir':'../data/imgs',
        'label_dir':'../data/labels',
        'model_dir':'../models/curr/',
        'display_iters':10,
        'test_iters':100,
        'epochs':1
    }

    current_iter = 0
    while (current_iter < params['max_iters']):
        current_iter = train(params)
        test(params)

def train(params):
    tf.reset_default_graph()

    Dataset.NUM_REPEAT = params['epochs']
    dataset = Dataset(params['data_dir'],
                      params['label_dir'],
                      batch_size=params['batch_size'])
    iters = fetch_iters()

    learning_rate = tf.train.exponential_decay(0.001, iters, 5000, 0.96)
    opt = tf.train.AdamOptimizer(learning_rate)

    def optimizer(loss):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads = opt.compute_gradients(loss)
        with tf.name_scope('clipping'):
            clipped_grads = []
            for grad, var in grads:
                if grad is not None:
                    grad = tf.clip_by_value(grad, -1.0, 1.0)
                else:
                    log.warn('No grad for {}'.format(var.name))

                clipped_grads.append((grad, var))

        return opt.apply_gradients(grads, global_step=iters)

    def train_metrics(logits, y):
        with tf.name_scope('metrics'):
            y = tf.reshape(y, [-1,Dataset.NUM_LABELS])
            logits = tf.reshape(logits, [-1,Dataset.NUM_LABELS])

            evaluate.evaluate_tensorboard(logits, y)

            return logits, y

    Model.DATA_FORMAT = params['data_format']
    model = Model(dataset, True)
    model.optimizer = optimizer
    model.train_metrics = train_metrics

    train = model.train()
    summaries = tf.summary.merge_all()

    # start training
    log.info('Training...')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with configured_session() as sess:
        if len(os.listdir(params['model_dir'])) > 0:
            latest_checkpoint = tf.train.latest_checkpoint(params['model_dir'])
            log.info('Restoring checkpoint {}'.format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
        else:
            log.info('Initializing Variables')
            sess.run(init)

        writer = tf.summary.FileWriter(params['train_dir'], graph=sess.graph)

        for _ in range(params['test_iters']):
            current_iter = iters.eval()
            log.info('ITER::{}'.format(current_iter))

            try:
                if current_iter % 10 == 0:
                    log.info('Evaluating...')
                    _, s = sess.run([train, summaries])
                    writer.add_summary(s, current_iter)
                    log.info('Saving')
                    save_name = '{}.ckpt'.format(Model.NAME)
                    saver.save(sess, os.path.join(params['model_dir'], save_name))
                else:
                    sess.run([train])
            except tf.errors.OutOfRangeError:
                log.info('Training Iter Complete')
                log.info('Saving')
                saver.save(sess, params['model_dir'], global_step=iters)
                break

        return current_iter

def test(params):
    tf.reset_default_graph()
    dataset = Dataset(params['data_dir'],
                      params['label_dir'],
                      batch_size=params['batch_size'])
    iters = fetch_iters()

    def test_metrics(logits, y):
        with tf.name_scope('metrics'):
            y = tf.reshape(y, [-1,Dataset.NUM_LABELS])
            logits = tf.reshape(logits, [-1,Dataset.NUM_LABELS])

            evaluate.evaluate_tensorboard(logits, y)

            return logits, y

    Model.DATA_FORMAT = params['data_format']
    model = Model(dataset, False)
    model.test_metrics = test_metrics

    test, metrics = model.test()
    summaries = tf.summary.merge_all()

    # restore graph
    log.info('Testing...')
    saver = tf.train.Saver()
    with configured_session() as sess:
        latest_checkpoint = tf.train.latest_checkpoint(params['model_dir'])
        log.info('Restoring checkpoint {}'.format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

        writer = tf.summary.FileWriter(params['test_dir'], graph=sess.graph)

        _, s = sess.run([test, summaries])

        writer.add_summary(s, iters)

if __name__=='__main__':
    main()