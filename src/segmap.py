# -*- coding: utf-8 -*-
# create twos sets segmaps based on each filter given an img_id
# 1st segmap is based on the original image (orig)
# 2nd segmap is based on a gaussian blur sigma=2 (blur)
#
# The segmaps in the output are labeled with ORIG and BLUR
#
# based on : # http://scikit-image.org/docs/dev/auto_examples/applications/plot_coins_segmentation.html
#


import numpy as np
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import sobel
from skimage import morphology

# helpers ----------------------------------------------------
def hist_dict(img):
    y, x = np.histogram(img, bins=100)
    return {'x':x, 'y':y}

def get_split_idx(hist_y):
    hist_diff = np.diff(hist_y)
    
    downhill = False
       
    for i in range(len(hist_diff)):
        val = hist_diff[i]
        
        if downhill == False:
            if val < 0:
                downhill = True
        else:
            if val >= 0:
                return i

# ------------------------------------------------------------

img_dir = '../data/jeyhan/gds2/GDS_deep2_1/'
output_dir = '../data/'

files = {'img_z': 'GDS_deep2_{}_z.fits',
         'img_v': 'GDS_deep2_{}_v.fits',
         'img_j': 'GDS_deep2_{}_j.fits',
         'img_h': 'GDS_deep2_{}_h.fits'}

orig_img_data = {}
blur_img_data = {}

orig_maps = {}
blur_maps = {}

orig_hists = {}
blur_hists = {}

orig_markers = {}
blur_markers = {}

img_id = 3518

for f_type in files.iterkeys():
    # get image data  
    file_dir = img_dir + files[f_type].format(img_id)
    orig = fits.getdata(file_dir)
    blur = gaussian_filter(orig, sigma=2)
    
    orig_img_data[f_type] = orig
    blur_img_data[f_type] = blur
    
    # get sorbel gradient map
    orig_map = sobel(orig)
    orig_maps[f_type] = orig_map
    
    blur_map = sobel(blur)
    blur_maps[f_type] = blur_map
    
    # get histogram
    orig_hist = hist_dict(orig)
    orig_hists[f_type] = orig_hist
    
    blur_hist = hist_dict(blur)
    blur_hists[f_type] = blur_hist
    
    orig_split_idx = get_split_idx(orig_hist['y'])
    blur_split_idx = get_split_idx(blur_hist['y'])    
    
    # get markers
    o_x = orig_hist['x']
    orig_marker = np.zeros_like(orig)
    orig_marker[orig < o_x[orig_split_idx]] = 1
    orig_marker[orig > o_x[orig_split_idx + 1]] = 2
    
    orig_markers[f_type] = orig_marker

    b_x = blur_hist['x']
    blur_marker = np.zeros_like(orig)
    blur_marker[blur < b_x[blur_split_idx]] = 1
    blur_marker[blur > b_x[blur_split_idx + 1]] = 2
    
    blur_markers[f_type] = blur_marker  
    
    orig_segmap = morphology.watershed(orig_map, orig_marker)
    blur_segmap = morphology.watershed(blur_map, blur_marker)
    
     # save new file
    orig_file_dir = output_dir + 'ORIG_NEW_SEG-' + files[f_type].format(img_id)
    blur_file_dir = output_dir + 'BLUR_NEW_SEG-' + files[f_type].format(img_id)
                                  
    fits.PrimaryHDU(orig_segmap).writeto(orig_file_dir)
    fits.PrimaryHDU(blur_segmap).writeto(blur_file_dir)
    
    