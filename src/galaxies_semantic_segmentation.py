import os

import numpy as np
import pandas as pd
from astropy.io import fits

import tensorflow as tf

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

        with_path = lambda i: os.path.join(img_dir, i)
        train_x = np.array([with_path(img) for img in all_imgs[:split_idx]])
        test_x = np.array([with_path(img) for img in all_imgs[split_idx:]])

        train_y, train_segmap = zip(*Dataset._get_img_labels(labels_dir, train_x))
        train_y, train_segmap = np.array(train_y), np.array(train_segmap)

        test_y, test_segmap = zip(*Dataset._get_img_labels(labels_dir, test_x))
        test_y, test_segmap = np.array(test_y), np.array(test_segmap)

        #print(train_x.shape, train_y.shape)

        self.batch_size = batch_size
        self.train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y, train_segmap))
        self.test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y, test_segmap))

    @staticmethod
    def _get_img_labels(labels_dir, img_list):
        lbl_columns = ['ClSph','ClDk','ClIr','ClPS','ClUn']

        label_file = os.path.join(labels_dir, 'labels.csv')
        labels = pd.read_csv(label_file)
        

        label_list = []
        for i in img_list:
            i = i.split('/')[-1]
            lbl_id = 'GDS_{}'.format(i.replace('.fits', ''))
            lbl = labels.loc[labels['ID']==lbl_id, lbl_columns].values.flatten()

            s_file = 'GDS_{}_segmap.fits'.format(i.replace('.fits', ''))
            s_file = os.path.join(labels_dir, 'segmaps', s_file)

            label_list.append((lbl.astype(np.float32), s_file))

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
    def tf_prep_input(x, y, segmap):
        x, y = tf.py_func(Dataset.prep_input, [x, y, segmap], (tf.float32, tf.float32))
        x.set_shape([Dataset.IMG_IN, Dataset.IMG_IN, Dataset.IN_CHANNELS])
        y.set_shape([Dataset.IMG_IN, Dataset.IMG_IN, Dataset.NUM_LABELS])
        return x, y

    @staticmethod
    def prep_input(x_file, lbl, segmap_file):
        x_file, segmap_file = str(x_file, 'utf-8'), str(segmap_file, 'utf-8')

        x = Dataset._safe_fits_open(x_file).astype(np.float32)
        segmap = Dataset._safe_fits_open(segmap_file)
        img_id = int(segmap_file.split('_')[-2])

        y = np.zeros([Dataset.IMG_IN, Dataset.IMG_IN, Dataset.NUM_LABELS], dtype=np.float32)
        y[segmap==img_id] = lbl
        y[segmap!=img_id] = Dataset.BACKGROUND

        print(x.shape, y.shape)

        return [x, y]

    @staticmethod
    def _safe_fits_open(fits_file):
        f = fits.getdata(fits_file)
        img = f.copy()
        del f
        return img

    @staticmethod 
    def preprocess_train(x, y):
        return Dataset._preprocess_data(x, y, True)

    @staticmethod
    def preprocess_test(x, y):
        return Dataset._preprocess_data(x, y, False)

    @staticmethod
    def _preprocess_data(x, y, is_training):
        t = tf.concat([x, y], axis=-1)
        
        if is_training:
            t = Dataset._augment(t)

        t = Dataset._crop(t, is_training)

        x, y = t[:,:,0,tf.newaxis], t[:,:,1:]
            
        return (Dataset._stadardize(x), y)

    @staticmethod
    def _augment(t):
        angle = tf.random_uniform([1], maxval=360)
        t = tf.contrib.image.rotate(t, angle, interpolation='BILINEAR')
        t = tf.image.random_flip_left_right(t)
        t = tf.image.random_flip_up_down(t)

        return t

    @staticmethod
    def _stadardize(x):
        x = tf.image.per_image_standardization(x)
        x = tf.reduce_mean(x, axis=2)

        return x

    @staticmethod
    def _crop(t, is_training):
        if is_training:
            pad_v = Dataset.IMG_IN + Dataset.PRE_PAD
            out_shape = [Dataset.IMG_OUT, Dataset.IMG_OUT, Dataset.IN_CHANNELS]
            t = tf.image.resize_image_with_crop_or_pad(t, pad_v, pad_v)
            t = tf.random_crop(t, out_shape)
        else:
            t = tf.image.central_crop(t, Dataset.IMG_OUT/Dataset.IMG_IN)

        return t


def main():
    data_dir = '../data/imgs'
    label_dir = '../data/labels'

    dataset = Dataset(data_dir, label_dir)

    with tf.Session() as sess:
        print(sess.run(dataset.train.get_next()))
        
if __name__ == "__main__": main()
