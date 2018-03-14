import os
from tf_src import tf_logger as log
from galaxies_semantic_segmentation import Dataset
from resnet_semantic_segmentation import Model
from tf_src.utils import configured_session, fetch_iters
import tensorflow as tf

import evaluate

def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)

    # parse params
    DATA_FORMAT = 'channels_last'
    TRAIN_DIR = '../report/tf-log/train/'
    TEST_DIR = '../report/tf-log/test/'
    MAX_ITERS = 10000
    BATCH_SIZE = 64
    data_dir = '../data/imgs'
    label_dir = '../data/labels'
    model_dir = '../models/curr/'
    display_iters = 10
    test_iters = 100


    current_iter = 0
    while (current_iter < MAX_ITERS):
        # setup training graph
        tf.reset_default_graph()
        Dataset.NUM_REPEAT = 1
        dataset = Dataset(data_dir, label_dir, batch_size=BATCH_SIZE)
        iters = fetch_iters()

        opt = tf.train.AdamOptimizer(0.001)
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
                #evaluate.tensorboard_confusion_matrix(logits, y)

                return logits, y

        Model.DATA_FORMAT = 'channels_last'
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
            if len(os.listdir(model_dir)) > 0:
                latest_checkpoint = tf.train.latest_checkpoint(model_dir)
                log.info('Restoring checkpoint {}'.format(latest_checkpoint))
                saver.restore(sess, latest_checkpoint)
            else:
                log.info('Initializing Variables')
                sess.run(init)

            writer = tf.summary.FileWriter(TRAIN_DIR, graph=sess.graph)

            while iters.eval() < MAX_ITERS:
                current_iter = iters.eval()
                log.info('ITER::{}'.format(current_iter))

                try:
                    if current_iter % 10 == 0:
                        log.info('Evaluating...')
                        _, s = sess.run([train, summaries])
                        writer.add_summary(s, current_iter)
                        log.info('Saving')
                        saver.save(sess, model_dir)
                    else:
                        sess.run([train])
                except tf.errors.OutOfRangeError:
                    log.info('Training Iter Complete')
                    log.info('Saving')
                    saver.save(sess, model_dir)
                    break

        # destroy graph/session/saver
        tf.reset_default_graph()
        dataset = Dataset(data_dir, label_dir, batch_size=64)
        iters = fetch_iters()

        def test_metrics(logits, y):
            with tf.name_scope('metrics'):
                y = tf.reshape(y, [-1,Dataset.NUM_LABELS])
                logits = tf.reshape(logits, [-1,Dataset.NUM_LABELS])

                evaluate.evaluate_tensorboard(logits, y)

                return logits, y

        Model.DATA_FORMAT = 'channels_last'
        model = Model(dataset, False)
        model.test_metrics = test_metrics

        test, metrics = model.test()
        summaries = tf.summary.merge_all()

        # restore graph
        log.info('Testing...')
        saver = tf.train.Saver()
        with configured_session() as sess:
            latest_checkpoint = tf.train.latest_checkpoint(model_dir)
            log.info('Restoring checkpoint {}'.format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)

            writer = tf.summary.FileWriter(TEST_DIR, graph=sess.graph)

            _, s = sess.run([test, metrics])

            writer.add_summary(s, current_iter)


    # do it again, again, and again

if __name__=='__main__':
    # parse params here and perhaps setup train/test instead of all at once
    main()