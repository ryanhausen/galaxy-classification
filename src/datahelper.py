# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from random import shuffle, randint
from astropy.io import fits

from PIL import Image, ImageChops

from copy import deepcopy

class DataHelper(object):
    """
    Provides shuffling and batches for the neural net
    """    
    
    def __init__(self, 
                 batch_size=15, 
                 train_size=0.8, 
                 shuffle_train=True,
                 augment=True,
                 data_dir='../data'):
        self._batch_size = batch_size
        self._augment = augment
        self._shuffle = shuffle
        self._imgs_dir = os.path.join(data_dir, 'imgs')
        self._imgs_list = os.listdir(self._imgs_dir)
        self._lbls = pd.read_csv(os.path.join(data_dir,'labels/labels.csv'))
        self._noise_tbl = pd.read_csv(os.path.join(data_dir, 'noise/noise_range.csv'))
        self._lbl_cols = ['ClSph', 'ClDk', 'ClIr', 'ClPS', 'ClUn']
        self._num_classes = len(self._lbl_cols)
        self._idx = 0        
        self.training = True        
        self.testing = False        

        size_change = False        
        num_train_examples = int(len(self._imgs_list) * train_size)
        
        batch_train = num_train_examples % batch_size
        if batch_train != 0:
            size_change = True
            
            if batch_train > batch_size / 2:
                num_train_examples += batch_train
            else:
                num_train_examples -= batch_train
        
        if size_change:
            msg = 'Batch didnt divide evenly into training examples ' + \
                  ' adjusted training size from {} to {}'
            print msg.format(train_size, float(num_train_examples) /  float(len(self._imgs_list)))
        
        # we want to use the same test images every time so they are set 
        # aside before the shuffle
        self._train_imgs = self._imgs_list[:num_train_examples]        
        self._test_imgs = self._imgs_list[num_train_examples:]
        
        if len(self._train_imgs) % batch_size != 0:
            err = 'Batch size must divide evenly into training. Batch: {} Train size: {}'
            raise Exception(err.format(batch_size, len(self._train_imgs)))
        
        if shuffle_train:
            shuffle(self._train_imgs)
        
    def _augment_image(self, img,img_id):    
        # rewrite into a single affine transformation       
      
        flip_type = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        
        rotation = randint(0,360)
        x_shift = randint(-4,4)
        y_shift = randint(-4,4)
        flip = randint(0,1)        
        f_type = flip_type[randint(0,1)]
        
        bands = ['h','j','v','z']        
        
        tmp = []

        shp_rng = None
        
        try:
            shp_rng = img.shape[2]
        except Exception:
            raise Exception('{} invalid shape: {}'.format(img_id,np.shape(img)))

        for i in range(shp_rng):
            
            tmp_img = Image.fromarray(img[:,:,i])
            
            #rotation
            tmp_img = tmp_img.rotate(rotation)       
            
            # shift -4 to 4 for x and y
            tmp_img = ImageChops.offset(tmp_img, x_shift, y_shift)
            
            # Scaling to go here, Brant wants to avoid scaling if we can for now
            
            if flip:
                tmp_img = tmp_img.transpose(f_type)
            
            
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
            
            #fits.PrimaryHDU(np.array(cpy_img)).writeto('../data/imgs/{}.fits'.format((i+1)*'a'))            
            
            tmp.append(cpy_img)
        
        return np.dstack(tmp)
        
        return img
    
    def get_next_batch(self):
        if self.training:        
            end_idx = self._idx + self._batch_size
            sources = self._train_imgs[self._idx:end_idx]

            x = []
            y = []
            
            for s in sources:
                s_id = s[:-5]                  
                x_dir = os.path.join(self._imgs_dir,s)  
                
                x_tmp =  fits.getdata(x_dir)
                
                if self._augment:
                    x_tmp = self._augment_image(x_tmp, s_id)
                                                
                x.append(x_tmp)

                # for the labels we need to prefix GDS_
                s_id = 'GDS_' + s[:-5]                  
                lbl = self._lbls.loc[self._lbls['ID']==s_id, self._lbl_cols]
                y.append(lbl.values.reshape(self._num_classes))
            
            x = np.array(x)
            y = np.array(y)       
            
            if end_idx >= len(self._train_imgs):
                self.training = False
                self.testing = True
                self._idx = 0
            else:
                self._idx = end_idx
            
            return (x, y)
        else:
            end_idx = self._idx + self._batch_size
            sources = self._test_imgs[self._idx:end_idx]
                        
            x = []
            y = []
            
            for s in sources:
                x_dir = os.path.join(self._imgs_dir,s)          
                
                x.append(fits.getdata(x_dir))
                
                s_id = 'GDS_' + s[:-5]                
                lbl = self._lbls.loc[self._lbls['ID']==s_id, self._lbl_cols]
                y.append(lbl.values.reshape(self._num_classes))
            
            x = np.array(x)
            y = np.array(y)       
            
            if end_idx + self._batch_size >= len(self._test_imgs):            
                self.testing = False
            else:
                self._idx = end_idx
            
            return (x, y)
            
            
    # Helper 
    # http://stackoverflow.com/a/5295202
    def _scale_to(self, x, rng, minmax):       
        a, b = rng
        mn, mx = minmax
    
        return (((b-a)*(x-mn))/(mx-mn)) + a