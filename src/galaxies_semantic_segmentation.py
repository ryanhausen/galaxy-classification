import os

import pandas as pd

import tensorflow as tf
import tensorflow.data as tfd

class Dataset:
    IMG_IN = 84 
    PRE_PAD = 10
    IMG_OUT = 80
    IN_CHANNELS = 4
    def __init__(self, img_dir, labels_dir, split=0.8, batch_size=25):
        all_imgs = os.listdir(img_dir)
        split_idx = int(len(all_imgs) * split)

        train_x = all_imgs[:split_idx]
        test_x = all_imgs[split_idx:]

        labels = pd.read_csv(labels_dir)

        train_y = Dataset._get_img_labels(labels, train_x)
        test_y = Dataset._get_img_labels(labels, test_x)

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

    def train():
        None

    def test():
        None

    @staticmethod
    def _preprocess_x(x, is_training):
        if is_training:
            pad_v = Dataset.IMG_IN + Dataset.PRE_PAD
            out_shape = [Dataset.IMG_OUT, Dataset.IMG_OUT, Dataset.IN_CHANNELS]
            angle = tf.random_uniform(1, maxval=360)

            x = tf.image.resize_image_with_crop_or_pad(x, pad_v, pad_v)
            x = tf.random_crop(x, out_shape)

            
            x = tf.contrib.image.rotate(x, angle, interpolation='BILINEAR')
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_flip_up_down(x)


        
        x = tf.image.per_image_standardization(x)
        x = tf.reduce_mean(x, axis=2)
        return x

    @staticmethod
    def _preprocess_y(y):
        None
