# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from random import shuffle
from astropy.io import fits

class DataHelper(object):
    """
    Provides shuffling and batches for the neural net
    """    
    
    def __init__(self, 
                 batch_size=15, 
                 test_size=0.2, 
                 shuffle_train=True, 
                 data_dir='../data'):
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._imgs_dir = os.path.join(data_dir, 'imgs')
        self._imgs_list = os.listdir(self._imgs_dir)
        self._lbls = pd.read_csv(os.path.join(data_dir,'labels/labels.csv'))
        self._idx = 0        
        
        num_test_examples = int(len(self._imgs_list) * test_size)
        
        # we want to use the same test images every time so they are set 
        # aside before the shuffle
        self._test_imgs = self._imgs_list[:(-1 * num_test_examples)]
        self._train_imgs = self._imgs_list[:num_test_examples]        
                
        if shuffle_train:
            shuffle(self._train_imgs)
        
        
    def get_next_batch(self, train=True):
        if train:        
            end_idx = self._idx + self._batch_size
            sources = self._train_imgs[self._idx:end_idx]
            return np.array([fits.getdata(i) for i in sources])
        