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
        self._lbl_cols = ['ClSph', 'ClDk', 'ClIr', 'ClPS', 'ClUn']
        self._idx = 0        
        self.training = True        
        self.testing = False        
        
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
            
            print sources            
            
            x = [fits.getdata(os.path.join(self._imgs_dir,i)) for i in sources]
            for i in x:
                print 'type: {}, shape: {}'.format(type(i), np.shape(i))
            x = np.array(x)
            y = np.array([self._lbls.loc[self._lbls['ID']== ('GDS_'+i[:-4]),self._lbl_cols] for i in sources])
            
            print np.shape(x)            
            print np.shape(y)            
            
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
                        
            x = np.array([fits.getdata(os.path.join(self._imgs_dir,i)) for i in sources])
            y = np.array([self._lbls.loc['ID'== ('GDS_'+i),self._lbl_cols].data for i in sources])
            
            if end_idx >= len(self._test_imgs):            
                self.testing = False
            else:
                self._idx = end_idx
            
            return (x, y)
            
    # TODO reset training method