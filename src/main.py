# -*- coding: utf-8 -*-

# imports
import numpy as np
from numpy.random import randint
from astropy.io import fits
from segmap import transform_segmap
from copy import deepcopy

# FLAGS
# set to true to replace all pixels except central source
# set to false to replace only the pixels in non-central source segmaps
replace_all_pixels = True
# END FLAGS


# get image and segmap
img_dir = '../data/jeyhan/gds2/GDS_deep2_1/'
output_dir = '../data/'

files = {'segmap':'GDS_deep2_{}_segmap.fits',
         'z': 'GDS_deep2_{}_z.fits',
         'v': 'GDS_deep2_{}_v.fits',
         'j': 'GDS_deep2_{}_j.fits',
         'h': 'GDS_deep2_{}_h.fits'}
img_data = {}

img_id = 3882

for f_type in files.iterkeys():
    file_dir = img_dir + files[f_type].format(img_id)
    img_data[f_type] = fits.getdata(file_dir)

transformed_segmap = transform_segmap([img_data['z'],img_data['v'],img_data['j'],img_data['h']],
                                     img_data['segmap'])



# get non source pixel locations
non_src_mask = img_data['segmap'] == 0
transformed_mask = transformed_segmap == 0

for img_key in img_data.iterkeys():
    if img_key != 'segmap':
        img = img_data[img_key]     
        noise = img[non_src_mask]  
        len_noise = len(noise)
        
        trans_img = deepcopy(img)               
        trans_noise = trans_img[transformed_mask]
        len_trans_noise = len(trans_noise)
        
        
        rpl_src_mask = None
        trans_rpl_src_mask = None
        if replace_all_pixels:
            rpl_src_mask = img_data['segmap'] != img_id
            trans_rpl_src_mask = transformed_segmap != img_id
        else:
            rpl_src_mask = np.logical_and(img_data['segmap'] > 0,
                                          img_data['segmap'] != img_id)
            
            trans_rpl_src_mask = np.logical_and(transformed_segmap > 0,
                                                transformed_segmap != img_id)
        
        for i in xrange(img.shape[0]):
            for j in xrange(img.shape[1]):
                if rpl_src_mask[i,j]:
                    img[i,j] = noise[randint(len_noise)]                    
                  
                if trans_rpl_src_mask[i,j]:
                    trans_img[i,j] = trans_noise[randint(len_trans_noise)]
                  
                  
                
        file_dir = output_dir + 'ORIG_' + files[img_key].format(img_id)
         
        hdu = fits.PrimaryHDU(img)
        hdu.writeto(file_dir)
        
        trans_dir = output_dir + 'NEW_' + files[img_key].format(img_id)       
        
        hdu = fits.PrimaryHDU(trans_img)
        hdu.writeto(trans_dir)

seg_dir = output_dir + 'transformed_segmap'
fits.PrimaryHDU(transformed_segmap).writeto(seg_dir)
