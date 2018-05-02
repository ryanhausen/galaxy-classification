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
        'epochs':1,
        'xentropy_coefficient':1,
        'dice_coefficient':1,
        'iou_thresholds':[0.9, 0.8, 0.7, 0.6]
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

    # https://arxiv.org/pdf/1701.08816.pdf eq 3, 4, 7, and 8
    def loss_func(logits, y):
        # softmax loss, per pixel
        flat_logits = tf.reshape(logits, [-1, 5])
        flat_y = tf.reshape(y, [-1, 5])
        xentropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                  labels=flat_y)

        # class coeffecients, different from the paper above we don't have
        # one-hot classes per pixel and so instead of a hard count, we'll
        # used an expected pixel account
        dominant_class = tf.argmax(flat_y, axis=1, output_type=tf.int32)
        p_dominant_class = tf.reduce_max(flat_y, axis=1)

        class_coefficient = tf.zeros_like(xentropy_loss)
        for output_class_idx in range(5):
            class_pixels = tf.cast(tf.equal(output_class_idx, dominant_class),
                                   tf.float32)
            coef = tf.reduce_mean(class_pixels * p_dominant_class)
            class_coefficient = tf.add(class_coefficient, coef*class_pixels)

        class_coefficient = 1 / class_coefficient

        weighted_xentropy_loss = tf.reduce_mean(xentropy_loss * class_coefficient)

        # dice loss
        if params['data_format']=='channels_first':
            yh_background = tf.nn.sigmoid(logits[:,-1,:,:])
            y_background = y[:,-1,:,:]
        else:
            yh_background = tf.nn.sigmoid(logits[:,:,:,-1])
            y_background = y[:,:,:,-1]

        dice_numerator = tf.reduce_sum(y_background * yh_background,
                                       axis=[1,2])
        dice_denominator = tf.reduce_sum(y_background + yh_background,
                                         axis=[1,2])

        dice_loss = tf.reduce_mean(2 * dice_numerator / dice_denominator)

        total_loss = params['xentropy_coefficient'] * weighted_xentropy_loss
        total_loss = total_loss + params['dice_coefficient'] * (1-dice_loss)

        return total_loss

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


    running_metrics = dict()
    def train_metrics(logits, y):
        with tf.name_scope('metrics'):
            if params['data_format']=='channels_last':
                lbls = tf.reshape(y[:,:,:,-1], [-1])
                predictions = tf.reshape(tf.nn.softmax(logits)[:,:,:,-1], [-1])
            else:
                lbls = tf.reshape(y[:,-1,:,:], [-1])
                predictions = tf.reshape(tf.nn.softmax(logits, axis=1)[:,-1,:,:], [-1])

            for threshold in params['iou_thresholds']:
                preds = tf.cast(tf.greater_equal(predictions, threshold), tf.int32)
                name = 'iou-{}'.format(threshold)
                mean_iou, mean_iou_update = tf.metrics.mean_iou(lbls,
                                                                preds,
                                                                2,
                                                                name=name)

                running_metrics[name] = mean_iou_update
                tf.summary.scalar('mean-iou-{}'.format(threshold), mean_iou)

            flat_y = tf.reshape(y, [-1,Dataset.NUM_LABELS])
            flat_logits = tf.reshape(logits, [-1,Dataset.NUM_LABELS])
            xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=flat_y,
                                                                  logits=flat_logits)

            tf.summary.scalar('cross-entropy', tf.reduce_mean(xentropy))

            return logits, y

    Model.DATA_FORMAT = params['data_format']
    model = Model(dataset, True)
    model.optimizer = optimizer
    model.train_metrics = train_metrics
    model.loss_func = loss_func

    train = model.train()
    summaries = tf.summary.merge_all()

    metric_vars = []
    for name in running_metrics.keys():
        metric_vars.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                             scope='metrics/{}'.format(name)))
    metric_reset = tf.variables_initializer(metric_vars)
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
                    sess.run(metric_reset)
                    run_ops = [train] + list(running_metrics.values())
                    sess.run(run_ops)
                    s = sess.run(summaries)
                    writer.add_summary(s, current_iter)
                    log.info('Saving')
                    save_name = '{}.ckpt'.format(Model.NAME)
                    saver.save(sess,
                               os.path.join(params['model_dir'], save_name),
                               global_step=iters)
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

    running_metrics = dict()
    def test_metrics(logits, y):
        with tf.name_scope('metrics'):
            if params['data_format']=='channels_last':
                lbls = tf.reshape(y[:,:,:,-1], [-1])
                predictions = tf.reshape(tf.nn.softmax(logits)[:,:,:,-1], [-1])
            else:
                lbls = tf.reshape(y[:,-1,:,:], [-1])
                predictions = tf.reshape(tf.nn.softmax(logits, axis=1)[:,-1,:,:], [-1])

            for threshold in params['iou_thresholds']:
                preds = tf.cast(tf.greater_equal(predictions, threshold), tf.int32)
                name = 'iou-{}'.format(threshold)
                mean_iou, mean_iou_update = tf.metrics.mean_iou(lbls,
                                                                preds,
                                                                2,
                                                                name=name)

                running_metrics[name] = mean_iou_update
                tf.summary.scalar('mean-iou-{}'.format(threshold), mean_iou)

            flat_y = tf.reshape(y, [-1,Dataset.NUM_LABELS])
            flat_logits = tf.reshape(logits, [-1,Dataset.NUM_LABELS])
            class_labels = tf.argmax(flat_y)
            class_preds = tf.argmax(flat_logits)
            acc, acc_update = tf.metrics.mean_per_class_accuracy(class_labels,
                                                                 class_preds,
                                                                 5)
            running_metrics['acc'] = acc_update

            #morphs =['Spheroid', 'Disk', 'Irregular', 'Point Source', 'Background']
            #for i, m in enumerate(morphs):
            #    tf.summary.scalar('accuracy-{}'.format(m), acc[i])
            tf.summary.scalar('accuracy-class', acc)

            return logits, y

    Model.DATA_FORMAT = params['data_format']
    model = Model(dataset, False)
    model.test_metrics = test_metrics

    test, _ = model.test()
    summaries = tf.summary.merge_all()

    metric_reset = tf.variables_initializer(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics'))

    # restore graph
    log.info('Testing...')
    saver = tf.train.Saver()
    with configured_session() as sess:
        latest_checkpoint = tf.train.latest_checkpoint(params['model_dir'])
        log.info('Restoring checkpoint {}'.format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
        sess.run(metric_reset)

        writer = tf.summary.FileWriter(params['test_dir'], graph=sess.graph)

        # go through the whole test set
        try:
            run_ops = [test] + list(running_metrics.values())
            while True:
                sess.run([run_ops])
        except tf.errors.OutOfRangeError:
            pass

        s = sess.run(summaries)
        writer.add_summary(s, iters.eval())

if __name__=='__main__':
    main()