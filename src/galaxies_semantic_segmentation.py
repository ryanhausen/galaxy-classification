import os

import numpy as np
import pandas as pd
from astropy.io import fits

import tensorflow as tf
import tensorflow.data as tfd

class Dataset:
    IMG_IN = 84 
    PRE_PAD = 10
    IMG_OUT = 80
    IN_CHANNELS = 4
    NUM_LABELS = 5
    BACKGROUND = np.array([0, 0, 0, 0, 1], dtype=np.float32)

    def __init__(self, img_dir, labels_dir, split=0.8, batch_size=25):
        all_imgs = os.listdir(img_dir)
        split_idx = int(len(all_imgs) * split)

        train_x = all_imgs[:split_idx]
        test_x = all_imgs[split_idx:]

        labels = pd.read_csv(labels_dir)

        train_y = Dataset._get_img_labels(labels, train_x)
        test_y = Dataset._get_img_labels(labels, test_x)

        self.batch_size = batch_size
        self.train_data = tfd.Dataset.from_tensor_slices((train_x, train_y))
        self.test_data = tfd.Dataset.from_tensor_slices((test_x, test_y))

    @staticmethod
    def _get_img_labels(labels_dir, img_list):
        lbl_columns = ['ClSph','ClDk','ClIr','ClPS','ClUn']

        labels = os.path.join(labels_dir, 'labels.csv')

        label_list = []
        for i in img_list:
            lbl_id = 'GDS_{}'.format(i.replace('.fits', ''))
            lbl = labels.loc[labels['ID']==lbl_id, lbl_columns].values.flatten()

            s_file = 'GDS_{}_segmap.fits'.format(i.replace('.fits', ''))
            os.path.join(labels_dir, 'segmaps', s_file)

            label_list.append((lbl, s_file))

        return label_list

    @property
    def train(self):
        training_data = self.train_data.map(Dataset.tf_prep_input)
        training_data = training_data.map(Dataset.preprocess_train)
        training_data = training_data.batch(self.batch_size)
        return training_data.make_one_shot_iterator()

        
    @property
    def test(self):
        training_data = self.train_data.map(Dataset.tf_prep_input)
        training_data = training_data.map(Dataset.preprocess_test)
        training_data = training_data.batch(self.batch_size)
        return training_data.make_one_shot_iterator()

    @staticmethod
    def tf_prep_input(dataset_item):
        return tf.py_func(Dataset.prep_input, dataset_item, tf.float32)

    @staticmethod
    def prep_input(data_in):
        x_file, label = data_in
        lbl, segmap_file = label

        x = Dataset._safe_fits_open(x_file)
        segmap = Dataset._safe_fits_open(segmap_file)
        img_id = int(segmap_file.split('_')[-2])

        y = np.zeros([Dataset.IMG_IN, Dataset.IMG_IN, Dataset.NUM_LABELS])
        y[segmap==img_id] = lbl
        y[segmap!=img_id] = Dataset.BACKGROUND

        return (x, y)

    @staticmethod
    def _safe_fits_open(fits_file):
        f = fits.getdata(fits_file)
        img = f.copy()
        del f
        return img

    @staticmethod 
    def preprocess_train(data_item):
        return Dataset._preprocess_data(data_item, True)

    @staticmethod
    def preprocess_test(data_item):
        return Dataset._preprocess_data(data_item, False)

    @staticmethod
    def _preprocess_data(data_item, is_training):
        x, y = data_item
        seed = tf.random_uniform(1, maxval=1e9, dtype=tf.float32)
        if is_training:
            x = Dataset._augment(x, seed)
            y = Dataset._augment(y, seed)
        
        x = Dataset._crop(x, is_training, seed=seed)
        y = Dataset._crop(y, is_training, seed=seed)
            
        return (Dataset._stadardize(x), y)

    @staticmethod
    def _augment(t, seed):
        angle = tf.random_uniform(1, maxval=360, seed=seed)
        t = tf.contrib.image.rotate(t, angle, interpolation='BILINEAR')
        t = tf.image.random_flip_left_right(t, seed=seed)
        t = tf.image.random_flip_up_down(t, seed=seed)

        return t

    @staticmethod
    def _stadardize(x):
        x = tf.image.per_image_standardization(x)
        x = tf.reduce_mean(x, axis=2)

        return x

    @staticmethod
    def _crop(t, is_training, seed=None):
        if is_training:
            pad_v = Dataset.IMG_IN + Dataset.PRE_PAD
            out_shape = [Dataset.IMG_OUT, Dataset.IMG_OUT, Dataset.IN_CHANNELS]
            t = tf.image.resize_image_with_crop_or_pad(t, pad_v, pad_v)
            t = tf.random_crop(t, out_shape, seed=seed)
        else:
            t = tf.image.central_crop(t, Dataset.IMG_OUT/Dataset.IMG_IN)

        return t


def __main__():
    data_dir = '~/Documents/astro_data/our_images'
    label_dir = '~/Documents/galaxy-classification/data/labels'

    dataset = Dataset(data_dir, label_dir)

    with tf.Session() as sess:
        print(sess.run(dataset.train.get_next()))