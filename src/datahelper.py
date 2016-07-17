# -*- coding: utf-8 -*-

import os
import re
import pandas as pd

class DataHelper(object):
    """
    Class to get data in a nn friendly format.


    """    
    
    def __init__(self, shuffle=True, data_dir='../data/'):
        self._shuffle = shuffle
        self._data_dir = data_dir        
        
        self._lbls = self._load_labels(os.path.join(data_dir,'labels/table3csv'))
        self._sources = self._distinct_sources(data_dir)
        
    def _load_labels(table3dat_file):
        cols = ['Depth','Area','ID','RAdeg','DEdeg','Seq','Hmag','NCl','Sph','Dk',
                'Ir','DS','DI','DSI','PS','UnCl','M','Int1','Int2','Comp','NoInt',
                'Int','C0P0','C1P0','C2P0','C0P1','C1P1','C2P1','C0P2','C1P2','C2P2',
                'Dbl','ImgQ','Unc','FlagV','Flagz','FlagJ','TArms','Db','Asym','Sp',
                'Bar','PSc','edge','face','Tp','Ch','DkD','Bg']
                
        labels = pd.read_csv(table3dat_file, sep=',', names=cols, index_col=False)
        
        return labels
        
    def _distinct_sources(self, data_dir):
        sources = {}

        for f in os.listdir(data_dir):
            new_dir = os.path.join(data_dir,f)
            if os.path.isdir(new_dir):
                sources.update(self._distinct_sources(new_dir))
            else:
                match = re.search('[a-z]+\d_\d+_', f)
                if match and match.group(0)[:-1] not in sources:
                    sources[match.group(0)[:-1]] = data_dir

        return sources

    # TODO methods for access