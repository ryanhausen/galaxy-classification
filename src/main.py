# -*- coding: utf-8 -*-
# this file tests dropping other sources from a file and occupying there
# space with gaussian noise fitted to the noise of the image.
# test files 
# GDS_deep2_3518_segmap.fits
# GDS_deep2_3518_z.fits
# GDS_deep2_3518_v.fits
# GDS_deep2_3518_j.fits
# GDS_deep2_3518_h.fits

# imports
import numpy as np
from numpy.random import normal, randint
from astropy.io import fits

# FLAGS
# set to true to replace all pixels except central source
# set to false to replace only the pixels in non-central source segmaps
replace_all_pixels = True

# set to true to draw pixel values from a assumed nice gaussian based on noise
# set to false to replace pixel values with ones randomly drawn from the noise
use_gaussian = True
# END FLAGS


# get image and segmap
img_dir = '../data/jeyhan/gds2/GDS_deep2_1/'
output_dir = '../data/'

files = {'segmap':'GDS_deep2_{}_segmap.fits',
         'img_z': 'GDS_deep2_{}_z.fits',
         'img_v': 'GDS_deep2_{}_v.fits',
         'img_j': 'GDS_deep2_{}_j.fits',
         'img_h': 'GDS_deep2_{}_h.fits'}
img_data = {}

img_id = 3518

for f_type in files.iterkeys():
    file_dir = img_dir + files[f_type].format(img_id)
    img_data[f_type] = fits.open(file_dir)[0].data

# get non source pixel locations
non_src_mask = img_data['segmap'] == 0

for img_key in img_data.iterkeys():
    if img_key != 'segmap':
        img = img_data[img_key]        
        noise = img[non_src_mask]
        
        mu = None, sigma = None, len_noise = None    
        if use_gaussian:
            mu = np.mean(noise)
            sigma = np.sqrt(np.var(noise))
        else:
            len_noise = len(noise)
        
        rpl_src_mask = None
        if replace_all_pixels:
            rpl_src_mask = img_data['segmap'] != img_id
        else:
            rpl_src_mask = np.logical_and(img_data['segmap'] > 0,
                                          img_data['segmap'] != img_id)
        
        for i in xrange(img.shape[0]):
            for j in xrange(img.shape[1]):
                if rpl_src_mask[i,j]:
                    if use_gaussian:                    
                        img[i,j] = normal(loc=mu, scale=sigma)
                    else:
                        img[i,j] = noise[randint(len_noise)]
                              
                                  
        # save new file
        file_dir = output_dir + files[img_key].format(img_id)
                                  
        hdu = fits.PrimaryHDU(img)
        hdu.writeto(file_dir)
