import os
from tf_src import tf_logger as log
from galaxies_semantic_segmentation import Dataset
from convnet_semantic_segmentation import Model
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
        'init_learning_rate':1e-6,
        'data_dir':'../data/imgs',
        'label_dir':'../data/labels',
        'model_dir':'../models/curr/',
        'display_iters':10,
        'test_iters':100,
        'epochs':1,
        'xentropy_coefficient':1,
        'dice_coefficient':1,
        'block_config':[2,2,4,8]
    }

    current_iter = 0
    while (current_iter < params['max_iters']):
        current_iter = train(params)
        test(params)

def eval_metrics(yh, y, metrics_dict):
    """
    yh: network output [n,h,w,c]
    y:  labels         [n,h,w,c]
    """

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    classes = ['spheroid', 'disk', 'irregular', 'point_source', 'background']

    with tf.name_scope('metrics'):
        # Cacluate the mean IOU for background/not background at each threshold
        with tf.name_scope('ious'):
            yh_bkg = tf.reshape(tf.nn.sigmoid(yh[:,:,:,-1]), [-1])
            y_bkg = tf.reshape(y[:,:,:,-1], [-1])
            for threshold in thresholds:
                name = 'iou-{}'.format(threshold)
                with tf.name_scope(name):
                    preds = tf.cast(tf.greater_equal(yh_bkg, threshold), tf.int32)
                    metric, update_op = tf.metrics.mean_iou(y_bkg, preds, 2, name=name)
                    metrics_dict[name] = update_op
                    tf.summary.scalar(name, metric)

        # Calculate the accuracy per class per pixel
        with tf.name_scope('accuracies'):
            y = tf.reshape(y, [-1, 5])
            yh = tf.reshape(yh, [-1, 5])
            lbls = tf.argmax(y, 1)
            preds = tf.argmax(yh, 1)
            for i, c in enumerate(classes):
                in_c = tf.equal(lbls, i)
                metric, update_op = tf.metrics.accuracy(lbls,
                                                        preds,
                                                        weights=in_c,
                                                        name=name)
                metrics_dict[name] = update_op
                tf.summary.scalar(name, metric)

    return metrics_dict


def train(params):
    tf.reset_default_graph()

    Dataset.NUM_REPEAT = params['epochs']
    dataset = Dataset(params['data_dir'],
                      params['label_dir'],
                      batch_size=params['batch_size'])
    iters = fetch_iters()

    learning_rate = tf.train.exponential_decay(params['init_learning_rate'], iters, 5000, 0.96)
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

        dice_loss = 1 - tf.reduce_mean(2 * dice_numerator / dice_denominator)

        total_loss = params['xentropy_coefficient'] * weighted_xentropy_loss
        total_loss = total_loss + params['dice_coefficient'] * (dice_loss)

        with tf.name_scope('loss'):
            tf.summary.scalar('Cross Entropy', tf.reduce_mean(xentropy_loss))
            tf.summary.scalar('Weighted Entropy', weighted_xentropy_loss)
            tf.summary.scalar('Dice Loss', dice_loss)
            tf.summary.scalar('Total Loss', total_loss)

        return total_loss

    def optimizer(loss):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients = opt.compute_gradients(loss)
        with tf.name_scope('clipping'):
            clipped = []
            with tf.name_scope('gradients'):
                for g, v in gradients:
                    g = tf.clip_by_value(g, -10, 10)
                    clipped.append((g, v))

        return opt.apply_gradients(clipped, global_step=iters)


    running_metrics = dict()
    def train_metrics(logits, y):
        if params['data_format']=='channels_first':
            _logits = tf.transpose(logits, [0,2,3,1])
            _y = tf.transpose(y, [0,2,3,1])
            eval_metrics(_logits, _y, running_metrics)
        else:
            eval_metrics(logits, y, running_metrics)

        return logits, y


    Model.DATA_FORMAT = params['data_format']
    Model.BLOCK_CONFIG = params['block_config']
    model = Model(dataset, True)
    model.optimizer = optimizer
    model.train_metrics = train_metrics
    model.loss_func = loss_func

    train = model.train()

    summaries = tf.summary.merge_all()

    metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics/*')
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
        if params['data_format']=='channels_first':
            _logits = tf.transpose(logits, [0,2,3,1])
            _y = tf.transpose(y, [0,2,3,1])
            eval_metrics(_logits, _y, running_metrics)
        else:
            eval_metrics(logits, y, running_metrics)

        return logits, y

    Model.DATA_FORMAT = params['data_format']
    model = Model(dataset, False)
    model.test_metrics = test_metrics

    test, _ = model.test()
    summaries = tf.summary.merge_all()


    metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics/*')
    metric_reset = tf.variables_initializer(metric_vars)

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
        writer.flush()
        writer.close()

if __name__=='__main__':
    main()