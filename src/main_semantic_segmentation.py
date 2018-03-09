from tf_src import tf_logger as log
from galaxies_semantic_segmentation import Dataset
from resnet_semantic_segmentation import Model
import tensorflow as tf

def main():

    # parse params
    DATA_FORMAT = 'channels_last'
    TRAIN_DIR = 'tf-log/train/'
    TEST_DIR = 'tf-log/test/'

    # define methods
    iters = tf.Variable(1, trainable=False)



    #setup graph/session/saver
    data_dir = '../data/imgs'
    label_dir = '../data/labels'

    log.info('Training...')
    tf.reset_default_graph()

    saver = tf.s
    dataset = Dataset(data_dir, label_dir, batch_size=64)

    opt = tf.train.MomentumOptimizer(0.001, 0.9)
    def optimizer(loss):
        opt.minimize(loss, global_step=iters)

    def train_metrics(yh, y):
        None

    def test_metrics(yh, y):
        None


    model = Model(dataset, True)




    # train

    # destroy graph/session/saver

    # setup graph/session/saver

    # restore graph

    # test

    # do it again, again, and again

if __name__=='__main__':
    main()