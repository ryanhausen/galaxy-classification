# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from random import shuffle
from astropy.io import fits

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
                 data_dir='../data'):
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._imgs_dir = os.path.join(data_dir, 'imgs')
        self._imgs_list = os.listdir(self._imgs_dir)
        self._lbls = pd.read_csv(os.path.join(data_dir,'labels/labels.csv'))
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
        
    def _augment_image(sef, img):
        None
    
    
    def get_next_batch(self):
        if self.training:        
            end_idx = self._idx + self._batch_size
            sources = self._train_imgs[self._idx:end_idx]

            x = []
            y = []
            
            for s in sources:
                x_dir = os.path.join(self._imgs_dir,s)          
                x.append(fits.getdata(x_dir).reshape(84,84,4))
                
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
                x.append(fits.getdata(x_dir).reshape(84,84,4))
                
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
            
    # TODO reset training method