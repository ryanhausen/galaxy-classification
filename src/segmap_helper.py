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
    min_val = np.min(hist_diff)    
        
    uphill = False    
    for i in range(len(hist_diff)):
        val = hist_diff[i]
        
        if val == min_val:
            uphill = True
        elif uphill and val >= 0:
            return i
                
# sobel images have to be between -1 and 1
# images outside this range tend to be the exception from my observation             
def _cap_at_1(img):
    img[img > 1] = 1
    img[img < -1] = -1
    return img
        
def _replace_overlapping_sources(coords, segmap, img_id):
    i, j = coords

    _check_middle_horz(segmap[i,:], j, img_id)
    _check_middle_vert(segmap[:,j], i, img_id)
    # TODO check for diaganols
        
def _check_middle_horz(a, j, img_id):
    val = a[j]
    
    right_cols = []    
    for i in range(j,len(a)):
        if a[i] > 0 and a[i] == val:
            right_cols.append(i)
        elif a[i] == img_id:
            break
        else:
            return None
    
    left_cols = []
    for i in reversed(range(0,j)):
        if a[i] > 0 and a[i] == val:
            left_cols.append(i)
        elif a[i] == img_id:
            break
        else:
            return None
        
    for col in (left_cols + right_cols):
        a[col] = img_id

def _check_middle_vert(a, i, img_id):
    val = a[i]
    
    right_cols = []
    reached_source = False    
    for j in range(i,len(a)):
        if a[j] > 0 and a[j] == val:
            right_cols.append(j)
        elif a[j] == img_id:
            reached_source = True
            break
        else:
            return None
    
    if reached_source == False:
        return None
    

    left_cols = []
    reached_source = False
    for j in reversed(range(0,i)):
        if a[j] > 0 and a[j] == val:
            left_cols.append(j)
        elif a[j] == img_id:
            reached_source = True
            break
        else:
            return None
    
    if reached_source == False:
        return None
    
    for col in (left_cols + right_cols):
        a[col] = img_id
    
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

    # TODO confirm that this logic will always be true
    img_id = segmap[dim1 / 2, dim2 / 2]
            
    # TODO integrate this into the loop above above to
    for i in range(dim1):
        vals = np.unique(union_segmap[i,:])
        # are there two sources in the row and is one of them the central source
        if len(vals) > 2 and img_id in np.unique(union_segmap[i,:]):
            for j in range(dim2):
                if union_segmap[i,j] > 0 and union_segmap[i,j] != img_id:
                    _replace_overlapping_sources((i,j), union_segmap, img_id)                

    return union_segmap
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    