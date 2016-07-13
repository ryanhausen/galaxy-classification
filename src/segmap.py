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
from math import hypot

# helpers ----------------------------------------------------
def hist_dict(img):
    y, x = np.histogram(img, bins=100)
    return {'x':x, 'y':y}

def get_split_idx(hist_y):
    hist_diff = np.diff(hist_y)
    max_val = np.max(hist_diff)    
    
    uphill = True    
    downhill = False
       
    for i in range(len(hist_diff)):
        val = hist_diff[i]
        
        if uphill == True:
            if val == max_val:
                uphill = False 
        elif downhill == False:
            if val < 0:
                downhill = True
        else:
            if val >= 0:
                return i
                
# for one of the libraries, i think sobel images have to be between -1 and 1
def cap_at_1(img):
    img[img > 1] = 1
    return img
        
# ------------------------------------------------------------

img_dir = '../data/jeyhan/gds2/GDS_deep2_1/'
output_dir = '../data/'

segmap_file = 'GDS_deep2_{}_segmap.fits'

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

orig_segmaps = {}
blur_segmaps = {}

# 3518
img_id = 3350

union_segmap = np.zeros((84,84))

for f_type in files.iterkeys():
    # get image data  
    file_dir = img_dir + files[f_type].format(img_id)
    orig = cap_at_1(fits.getdata(file_dir))    
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
    blur_marker = np.zeros_like(blur)
    blur_marker[blur < b_x[blur_split_idx]] = 1
    blur_marker[blur > b_x[blur_split_idx + 1]] = 2
    
    blur_markers[f_type] = blur_marker  
    
    orig_segmap = morphology.watershed(orig_map, orig_marker)
    blur_segmap = morphology.watershed(blur_map, blur_marker)   
    
     # save new file
    #orig_file_dir = output_dir + 'ORIG_NEW_SEG-' + files[f_type].format(img_id)
    blur_file_dir = output_dir + 'BLUR_NEW_SEG-' + files[f_type].format(img_id)
                                  
    #fits.PrimaryHDU(orig_segmap).writeto(orig_file_dir)
    fits.PrimaryHDU(blur_segmap).writeto(blur_file_dir)
    
    # union all blurs
    # union_segmap = cap_at_1(union_segmap + ((orig_segmap-1) + (blur_segmap-1)))
    union_segmap = cap_at_1(union_segmap + (blur_segmap-1))

# merge with original segmap
given_segmap = fits.getdata(img_dir + segmap_file.format(img_id))

for i in range(84):
    for j in range(84):
        # if the pixel has a value in the given segmap then give it that value
        if given_segmap[i,j] > 0:
            union_segmap[i,j] = given_segmap[i,j]
        # if our new segmap has a value that isn't in the given segmap then
        # find the closest label and assign it 
        elif union_segmap[i,j] > 0:
            # NAIVE IMPLEMENTATION FIND A BETTER ALGORITHM!!!!!
            coords = None
            min_dist = hypot(84,84)
            for k in range(84):
                for l in range(84):
                    if given_segmap[k,l] > 0 and hypot(i-k, j-l) < min_dist:
                        coords = (k,l)
                        min_dist = hypot(i-k, j-l)
                        
            union_segmap[i,j] = given_segmap[coords[0], coords[1]]
        
union_segmap[union_segmap == 1] = img_id
union_dir = output_dir + 'UNION_SEG_{}.fits'.format(img_id)
fits.PrimaryHDU(union_segmap).writeto(union_dir)