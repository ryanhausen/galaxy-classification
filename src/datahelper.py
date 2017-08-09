# -*- coding: utf-8 -*-

import os
import math
import pandas as pd
import numpy as np
from itertools import cycle
from random import shuffle, randint
from astropy.io import fits
from copy import deepcopy


from PIL import Image

class DataHelper(object):
    """
    Provides shuffling and batches for the neural net
    """

    def __init__(self,
                 batch_size=15,
                 train_size=0.8,
                 shuffle_train=True,
                 augment=True,
                 label_noise=None,
                 data_dir='../data',
                 bands = ['h','j','v','z'],
                 transform_func=None,
                 band_transform_func=None):
        """
             label_noise should be the stddev of the gaussian used to generate
             the noise
        """
        self._batch_size = batch_size
        self._augment = augment
        self._label_noise = label_noise
        self._shuffle = shuffle
        self._bands = bands
        self._transform_func = transform_func
        self._band_transform_func = band_transform_func
        self._drop_band = len(bands) < 4
        self._imgs_dir = os.path.join(data_dir, 'imgs')
        self._imgs_list = os.listdir(self._imgs_dir)
        self._lbls = pd.read_csv(os.path.join(data_dir,'labels/labels.csv'))
        self._noise_tbl = pd.read_csv(os.path.join(data_dir, 'noise/noise_range.csv'))
        self._lbl_cols = ['ClSph', 'ClDk', 'ClIr', 'ClPS', 'ClUn']
        self._num_classes = len(self._lbl_cols)
        self._idx = 0
        self.training = True
        self.testing = False

        # work in batches
        if batch_size:
            num_train_examples = int(len(self._imgs_list) * train_size)

#            batch_train = num_train_examples % batch_size
#            if batch_train != 0:
#
#                if batch_train > batch_size / 2:
#                    num_train_examples += batch_train
#                else:
#                    num_train_examples -= batch_train

                #msg = 'Batch didnt divide evenly into training examples ' + \
                #      ' adjusted training size from {} to {}'
                #print msg.format(train_size, float(num_train_examples) /  float(len(self._imgs_list)))

            # we want to use the same test images every time so they are set
            # aside before the shuffle
            self._train_imgs = self._imgs_list[:num_train_examples]
            self._build_iters(self._train_imgs, self._lbls)

            self._test_imgs = self._imgs_list[num_train_examples:]

#            if len(self._train_imgs) % batch_size != 0:
#                err = 'Batch size must divide evenly into training. Batch: {} Train size: {}'
#                raise Exception(err.format(batch_size, len(self._train_imgs)))

            if shuffle_train:
                shuffle(self._train_imgs)

        # one example at a time
        else:
            self.testing = True

    def _build_iters(self, img_file_names, lbls_df):

        self.train_count = 0
        self.train_iter = cycle(img_file_names)

        self.sph_list, self.sph_iter  = [], None
        self.dk_list, self.dk_iter  = [], None
        self.irr_list, self.irr_iter  = [], None
        self.ps_list, self.ps_iter  = [], None
        self.unk_list, self.unk_iter  = [], None

        for s in img_file_names:
            s_id = 'GDS_' + s[:-5]
            lbl = np.argmax(self._lbls.loc[self._lbls['ID']==s_id, self._lbl_cols].values)
            if lbl==0:
                self.sph_list.append(s)
            elif lbl==1:
                self.dk_list.append(s)
            elif lbl==2:
                self.irr_list.append(s)
            elif lbl==3:
                self.ps_list.append(s)
            elif lbl==4:
                self.unk_list.append(s)

        for coll in [self.sph_list,self.dk_list,self.irr_list,self.ps_list,self.unk_list]:
            shuffle(coll)

        self.sph_count,self.dk_count,self.irr_count,self.ps_count,self.unk_count = 0,0,0,0,0

        self.sph_iter = cycle(self.sph_list)
        self.dk_iter = cycle(self.dk_list)
        self.irr_iter = cycle(self.irr_list)
        self.ps_iter = cycle(self.ps_list)
        self.unk_iter = cycle(self.unk_list)


    def _augment_image(self, img, img_id):

        if self._transform_func:
            img = self._transform_func(img)

        rotation = randint(0,359)
        x_shift = randint(-4,4)
        y_shift = randint(-4,4)
        flip = randint(0,1)
        scale = math.exp(np.random.uniform(np.log(1.0 / 1.3), np.log(1.3)))
        brightness = np.random.normal(loc=0.0,scale=0.01)

        to_origin = DataHelper._translate(42,42)
        flipped = DataHelper._flip(randint(0,1)) if flip else np.identity(3)
        scaled = DataHelper._scale(scale)
        rotated = DataHelper._rotate(rotation)
        shifted = DataHelper._translate(x_shift, y_shift)
        recenter = DataHelper._translate(-42,-42)

        trans = to_origin.dot(flipped).dot(scaled).dot(rotated).dot(shifted).dot(recenter)
        trans = tuple(trans.flatten()[:6])

        bands = ['h','j','v','z']

        tmp = []

        shp_rng = None
        try:
            shp_rng = img.shape[2]
        except Exception:
            raise Exception('{} invalid shape: {}'.format(img_id,np.shape(img)))

        for i in range(shp_rng):
            if bands[i] in self._bands:
                tmp_img = img[:,:,i]

                if self._band_transform_func:
                    tmp_img = self._band_transform_func(tmp_img)

                tmp_img = Image.fromarray(tmp_img)

                tmp_img = tmp_img.transform((84,84), Image.AFFINE, data=trans, resample=Image.BILINEAR)

                tmp_img = np.asarray(tmp_img)
                noise = fits.getdata('../data/noise/{}.fits'.format(bands[i]))

                id_mask = (self._noise_tbl['ID']==img_id) & (self._noise_tbl['band']==bands[i])

                try:
                    rng = tuple(self._noise_tbl.loc[id_mask, ['mn', 'mx']].values[0])
                    noise = self._scale_to(noise, rng, (np.min(noise), np.max(noise)))
                except Exception:
                    None
                    # log this
                    #print 'unable to rescale noise for {}, {} likely there is no noise rescale from'.format(img_id, bands[i])

                len_noise = None
                if bands[i] not in ('h','j'):
                    noise = noise.flatten()
                    len_noise = len(noise)-1

                noise_mask = tmp_img == 0

                cpy_img = deepcopy(np.asarray(tmp_img))
                for j in range(cpy_img.shape[0]):
                    for k in range(cpy_img.shape[1]):
                        if noise_mask[j,k]:
                            if bands[i] in ('h','j'):
                                cpy_img[j,k] = noise[j,k]
                            else:
                                cpy_img[j,k] = noise[randint(0,len_noise)]

                tmp.append(cpy_img)

        if len(tmp) > len(self._bands):
            raise Exception('Image {} didnt augment properly'.format(img_id))

        return np.dstack(tmp)

        return img


    def img_name_server(self, fair_spread=True):

        if fair_spread:
            num_classes = 5

            img_names = []

            class_iters = [self.sph_iter,self.dk_iter,self.irr_iter,self.ps_iter,self.unk_iter]

            for i in range(self._batch_size//num_classes):
                for coll in class_iters:
                    img_names.append(next(coll))

            # TODO this will favor earlier classes, just choose %5==0 batch sizes
            for i in range(self._batch_size % num_classes):
                img_names.append(next(class_iters[i]))

            shuffle(img_names)

        else:
            img_names = [next(self.train_iter) for i in range(self._batch_size)]
            self.train_count += self._batch_size
            if self.train_count > len(self._train_imgs):
                self.train_count = 0
                shuffle(self._train_imgs)
                self.train_iter = cycle(self._train_imgs)

        return img_names

    def get_next_batch(self, include_ids=False, iter_based=False, force_test=False):
        if self.training and force_test==False:
            x, y = [], []

            if iter_based:
                sources = self.img_name_server()

            else:
                end_idx = self._idx + self._batch_size
                sources = self._train_imgs[self._idx:end_idx]

                if end_idx >= len(self._train_imgs):
                    self.training = False
                    self.testing = True
                    self._idx = 0
                else:
                    self._idx = end_idx


            for s in sources:
                s_id = s[:-5]
                x_dir = os.path.join(self._imgs_dir,s)

                try:
                    x_tmp = self._get_fits(x_dir, normalize=False)
                except Exception as e:
                    print(f'ERROR with {x_dir}')
                    raise e

                if self._augment:
                    x_tmp = self._augment_image(x_tmp, s_id)

                x.append(x_tmp.copy())
                del x_tmp

                # for the labels we need to prefix GDS_
                s_id = 'GDS_' + s[:-5]
                lbl = self._lbls.loc[self._lbls['ID']==s_id, self._lbl_cols]
                lbl = lbl.values.reshape(self._num_classes)

                if self._label_noise:
                    lbl_noise = np.random.normal(scale=self._label_noise,
                                                 size=self._num_classes)

                    # add noise and renormalize so we get a proper distribution
                    lbl = DataHelper._rescale_label(lbl + lbl_noise)

                y.append(lbl)

            x = np.array(x)
            y = np.array(y)

            if include_ids:
                return (sources, x, y)
            else:
                return (x, y)
        else:
            sources = self._test_imgs[:]
            bands = ['h','j','v','z']

            x = []
            y = []

            for s in sources:
                x_dir = os.path.join(self._imgs_dir,s)

                try:
                    raw =  self._get_fits(x_dir, normalize=False)
                except Exception as e:
                    print(f'ERROR with {x_dir}')
                    raise e

                if self._drop_band:
                    tmp = []
                    for i in range(4):
                        if bands[i] in self._bands:
                            tmp.append(raw[:,:,i])

                    raw = np.dstack(tmp)

                # to avoid os errors copy to a new mem location and
                # remove the file handle
                x.append(deepcopy(raw))
                del raw

                s_id = 'GDS_' + s[:-5]
                lbl = self._lbls.loc[self._lbls['ID']==s_id, self._lbl_cols]
                y.append(lbl.values.reshape(self._num_classes))

            x = np.array(x)
            y = np.array(y)

            self.training = True
            self.testing = False
            self._idx = 0

            if include_ids:
                return (sources, x, y)
            else:
                return (x, y)

    def get_class_distribution(self):
        cls_count_train = np.zeros([5,])
        cls_count_test = np.zeros([5,])


        for i in range(len(self._train_imgs)):
            src = self._train_imgs[i]
            s_id = 'GDS_' + src[:-5]
            lbl = self._lbls.loc[self._lbls['ID']==s_id, self._lbl_cols]
            lbl = lbl.values.reshape(self._num_classes)

            cls_count_train[np.argmax(lbl)] += 1

        for i in range(len(self._test_imgs)):
            src = self._test_imgs[i]
            s_id = 'GDS_' + src[:-5]
            lbl = self._lbls.loc[self._lbls['ID']==s_id, self._lbl_cols]
            lbl = lbl.values.reshape(self._num_classes)

            cls_count_test[np.argmax(lbl)] += 1

        return (cls_count_train, cls_count_test)

    def get_next_example(self, img_id=None):
        bands = ['h','j','v','z']

        source = None
        if img_id:
            source = img_id + '.fits'
            self.testing = False
        else:
            source = self._imgs_list[self._idx]
            self._idx += 1

        if self._idx >= len(self._imgs_list):
            self.testing = False

        x = fits.getdata(os.path.join(self._imgs_dir, source))

        tmp_x = []
        for i in range(4):
            if bands[i] in self._bands:
                tmp_x.append(x[:,:,i])

        x = np.dstack(tmp_x).reshape(1,84,84,len(self._bands))


        y = self._lbls.loc[self._lbls['ID']=='GDS_'+ source[:-5], self._lbl_cols]
        y = y.values.reshape(self._num_classes)

        return (x,y)




    def _get_fits(self, img_path, normalize=False):
        tmp = fits.getdata(img_path)

        if normalize:
            for i in range(tmp.shape[-1]):
                x = tmp[:,:,i]
                tmp[:,:,i] = (x-np.min(x))/(np.max(x)-np.min(x))

        return tmp

    # Helper
    @staticmethod
    def _rescale_label(y):
        return y / y.sum()

    # http://stackoverflow.com/a/5295202
    def _scale_to(self, x, rng, minmax):
        a, b = rng
        mn, mx = minmax

        return (((b-a)*(x-mn))/(mx-mn)) + a

    #http://stackoverflow.com/questions/34968722/softmax-function-python
    @staticmethod
    def _softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    @staticmethod
    def _translate(u,v):
        return np.array([
                [1.0, 0.0, float(u)],
                [0.0, 1.0, float(v)],
                [0.0, 0.0, 1.0]
            ])
    @staticmethod
    def _rotate(angle):
        angle = -math.radians(angle)
        cos = math.cos(angle)
        sin = math.sin(angle)

        return np.array([
                [cos, sin, 0.0],
                [-sin, cos, 0.0],
                [0.0, 0.0, 1.0]
            ])
    @staticmethod
    def _scale(v):
        return np.array([
                [v, 0.0, 0.0],
                [0.0, v, 0.0],
                [0.0, 0.0, 1.0]
            ])
    @staticmethod
    def _flip(flip_type):
        """
        Horizantle flip = 0
        Vertical flip = 1
        """
        if flip_type:
            return np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ])
        else:
            return np.array([
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ])
