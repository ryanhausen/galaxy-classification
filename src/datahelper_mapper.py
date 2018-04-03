import math
import os
from copy import deepcopy
from itertools import cycle
from random import randint, shuffle

import numpy as np
import pandas as pd
from astropy.io import fits
from PIL import Image

class DataHelper(object):
    def __init__(self,
                 train_size=0.8,
                 shuffle_train=True,
                 data_dir='../data',
                 out_size=None):
        self._shuffle_train = shuffle_train
        self.data_dir = data_dir
        self.out_size = out_size

        self._sources_dir = os.path.join(data_dir, 'imgs')
        self._segmaps_dir = os.path.join(data_dir, 'segmaps')
        self._sources = os.listdir(self._sources_dir)
        self._lbls = pd.read_csv(os.path.join(data_dir,'labels/labels.csv'))

        num_train = int(len(self._sources) * train_size)

        self._train_count = 0
        self._train_sources = self._sources[:num_train]
        self._test_sources = self._sources[num_train:]
        self._build_iters(self._train_sources)

    def next_batch(self, count:int, training:bool, even_class_dist:bool):
        lbl_columns = ['ClSph', 'ClDk', 'ClIr', 'ClPS']
        crop = self.out_size//2 if self.out_size else 22

        if training:
            x, y = [], []

            sources = self._source_name_server(count, even_dist=even_class_dist)

            count = 0
            for s in sources:
                s_id = s[:-5]

                # img ==========================================================
                try:
                    x_tmp = self._safe_fits(os.path.join(self._sources_dir,s))
                except Exception as e:
                    print('ERROR with {}')
                    raise e

                cx, cy = randint(38,42), randint(38,42)
                x_tmp = x_tmp[cy-crop:cy+crop, cx-crop:cx+crop]

                x_tmp = DataHelper._pre_process(x_tmp)
                # ==============================================================

                # label ========================================================
                s_id = 'GDS_' + s_id
                lbl = self._lbls.loc[self._lbls['ID']==s_id, lbl_columns]
                lbl = np.concatenate([lbl.values.reshape(len(lbl_columns)),
                                     np.zeros(1)])
                bkg_lbl = np.array([0, 0, 0, 0, 1])

                segmap = s_id + '_segmap.fits'
                segmap = self._safe_fits(os.path.join(self._segmaps_dir,segmap))
                segmap = segmap[cy-crop:cy+crop, cx-crop:cx+crop]
                x_tmp, segmap = self._augment(x_tmp, segmap)

                s_id = int(s_id.split('_')[-1])

                y_tmp = np.zeros(list(segmap.shape) + [5])
                y_tmp[segmap==s_id,:] = lbl
                y_tmp[segmap!=s_id,:] = bkg_lbl
                # ==============================================================
                x.append(x_tmp)
                y.append(y_tmp)

            return np.array(x), np.array(y)
        else:
            sources = self._test_sources
            x, y, = [], []

            for s in sources:
                s_id = s[:-5]
                try:
                    x_tmp = self._safe_fits(os.path.join(self._sources_dir,s))
                except Exception as e:
                    print('ERROR with {}')
                    raise e

                cx, cy = 42, 42
                x_tmp = x_tmp[cy-crop:cy+crop, cx-crop:cx+crop]

                x_tmp = DataHelper._pre_process(x_tmp)

                s_id = 'GDS_' + s_id
                lbl = self._lbls.loc[self._lbls['ID']==s_id, lbl_columns]
                lbl = np.concatenate([lbl.values.reshape(len(lbl_columns)),
                                     np.zeros(1)])
                bkg_lbl = np.array([0, 0, 0, 0, 1])

                segmap = s_id + '_segmap.fits'
                segmap = self._safe_fits(os.path.join(self._segmaps_dir,segmap))
                segmap = segmap[cy-crop:cy+crop, cx-crop:cx+crop]

                s_id = int(s_id.split('_')[-1])

                y_tmp = np.zeros(list(segmap.shape) + [5])
                y_tmp[segmap==s_id,:] = lbl
                y_tmp[segmap!=s_id,:] = bkg_lbl

                x.append(x_tmp)
                y.append(y_tmp)

            return np.array(x), np.array(y)

    def _build_iters(self, img_file_names):
        lbl_columns = ['ClSph', 'ClDk', 'ClIr', 'ClPS']


        self.train_count = 0
        self.train_iter = cycle(img_file_names)

        self.sph_list, self.sph_iter  = [], None
        self.dk_list, self.dk_iter  = [], None
        self.irr_list, self.irr_iter  = [], None
        self.ps_list, self.ps_iter  = [], None
        self.unk_list, self.unk_iter  = [], None

        for s in img_file_names:
            s_id = 'GDS_' + s[:-5]
            r_mask = self._lbls['ID']==s_id
            lbl = np.argmax(self._lbls.loc[r_mask, lbl_columns].values)

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

    #https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    @staticmethod
    def _pre_process(img):
        img = (img - img.mean()) / max(img.std(), 1/np.sqrt(np.prod(img.shape)))
        return np.mean(img, axis=2)[:,:,np.newaxis]

    def _augment(self, source, segmap):
        move = source.shape[0]//2
        rotation = randint(0, 359)
        flip = randint(0,1)
        scale = np.exp(np.random.uniform(1.0 / 1.3, np.log(1.3)))

        # source ===============================================================
        to_origin = DataHelper._translate(move, move)
        flipped = DataHelper._flip(randint(0, 1)) if flip else np.identity(3)
        scaled = DataHelper._scale(scale)
        rotated = DataHelper._rotate(rotation)
        recenter = DataHelper._translate(-move, -move)

        trans = to_origin.dot(flipped).dot(scaled).dot(rotated).dot(recenter)
        trans = tuple(trans.flatten()[:6])

        augmented = []
        for i in range(source.shape[2]):
            tmp = Image.fromarray(source[:,:,i])
            tmp = tmp.transform(source.shape[:2],
                                Image.AFFINE,
                                data=trans,
                                resample=Image.BILINEAR)

            augmented.append(deepcopy(np.asarray(tmp)))
        # ======================================================================

        # segmap ===============================================================
        trans = to_origin.dot(flipped).dot(rotated).dot(recenter)
        trans = tuple(trans.flatten()[:6])
        seg_tmp = Image.fromarray(segmap)
        seg_tmp = seg_tmp.transform(segmap.shape,
                                    Image.AFFINE,
                                    data=trans,
                                    resample=Image.NEAREST)
        segmap = np.asarray(seg_tmp)
        # ======================================================================


        return np.dstack(augmented), segmap

    def _source_name_server(self, count, even_dist=False):
        if even_dist:
            num_classes = 4
            img_names = []

            class_iters = [self.sph_iter,
                           self.dk_iter,
                           self.irr_iter,
                           self.ps_iter]

            for i in range(count//num_classes):
                for coll in class_iters:
                    img_names.append(next(coll))

            # TODO this will favor earlier classes, just choose %5==0 batch sizes
            for i in range(count % num_classes):
                img_names.append(next(class_iters[i]))

            shuffle(img_names)

        else:
            img_names = [next(self.train_iter) for i in range(count)]
            self.train_count += count
            if self.train_count > len(self._train_sources):
                self.train_count = 0
                shuffle(self._train_sources)
                self.train_iter = cycle(self._train_sources)

        return img_names



    @staticmethod
    def _safe_fits(img_path):
        tmp = fits.getdata(img_path)

        rtn_val = deepcopy(tmp)
        del tmp

        return rtn_val



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
