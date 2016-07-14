# -*- coding: utf-8 -*-
# based on : # http://scikit-image.org/docs/dev/auto_examples/applications/plot_coins_segmentation.html
#


import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import sobel
from skimage import morphology
from math import hypot

# helpers ----------------------------------------------------
def _get_split_idx(hist_y):
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
                
# for sobel images have to be between -1 and 1
def _cap_at_1(img):
    img[img > 1] = 1
    return img
        
# ------------------------------------------------------------


def transform_segmap(imgs, segmap, sigma=2, truncate=4.0):
    """
    imgs : list of img in different bands as numpy arrays 
    segmap : the segmap from which labels can be drawn
    sigma : passed as sigma to scipy.ndimage.filters.gaussian_filter     
    truncate: passed as truncate to scipy.ndimage.filters.gaussian_filter
    
    Uses Gaussian to blur input images and broaden passed in segmap
    """
    union_segmap = np.zeros_like(segmap)

    # create blurred segmaps  
    for img in imgs:
        img = gaussian_filter(_cap_at_1(img), sigma=sigma, truncate=truncate)
        img_map = sobel(img)
        y, x = np.histogram(img, bins=100)
        
        split_idx = _get_split_idx(y)
        markers = np.zeros_like(img)
        markers[img < x[split_idx]] = 1
        markers[img > x[split_idx + 1]] = 2
        
        new_segmap = morphology.watershed(img_map, markers)
        
        union_segmap = _cap_at_1(union_segmap + (new_segmap-1))

    # merge and label with passed in segmap
    dim1 = segmap.shape[0]
    dim2 = segmap.shape[1]

    for i in range(dim1):
        for j in range(dim2):
            # if the pixel has a value in the given segmap then give it that value
            if segmap[i,j] > 0:
                union_segmap[i,j] = segmap[i,j]
            # if our new segmap has a value that isn't in the given segmap then
            # find the closest label and assign it 
            elif union_segmap[i,j] > 0:
                # NAIVE IMPLEMENTATION FIND A BETTER ALGORITHM!!!!!
                coords = None
                min_dist = hypot(dim1,dim2)
                for k in range(dim1):
                    for l in range(dim2):
                        if segmap[k,l] > 0 and hypot(i-k, j-l) < min_dist:
                            coords = (k,l)
                            min_dist = hypot(i-k, j-l)
                            
                union_segmap[i,j] = segmap[coords[0], coords[1]]

    return union_segmap