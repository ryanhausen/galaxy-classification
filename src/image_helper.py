# python libs
from copy import deepcopy

# third party libs
import numpy as np
from numpy.random import randint
from scipy.ndimage.filters import convolve
from scipy.misc import imresize


def transform_image(img, band, segmap, tinytim):
    # Helper 
    # http://stackoverflow.com/a/5295202
    def scale_to(x, rng, minmax):       
        a, b = rng
        mn, mx = minmax
    
        return (((b-a)*(x-mn))/(mx-mn)) + a
    
    # The img id should be in the center of the segmap
    img_id = segmap[segmap.shape[0] / 2, segmap.shape[1] / 2]    
         
    # use this copy to house the noise we want to convolve with tiny tim
    noise = img[segmap == 0]
    len_noise = len(noise)   
    
    # if there isn't any noise in the image, the there isn't anything for us
    # to replace, wide2_9682 is an example of this
    if len_noise > 0:
        cpy_noise = deepcopy(img)    
        
        for i in xrange(img.shape[0]):
            for j in xrange(img.shape[1]): 
                if segmap[i,j] > 0:
                    cpy_noise[i,j] = noise[randint(len_noise)]
    
    
         # resize and rescale the tiny tim image
        tinytim = imresize(tinytim, (25,25))
        tinytim = scale_to(tinytim, (.001, 1), (np.min(tinytim), np.max(tinytim)))
    
        # convolve the noise only image with the modified tiny tim kernel
        tt_img = convolve(cpy_noise, tinytim)
    
        noise_rng = (np.min(noise), np.max(noise))
        tt_range =  (np.min(tt_img), np.max(tt_img))   
        
        # recale to match the noise of the original image
        tt_img = scale_to(tt_img, noise_rng, tt_range)
    
        # we want to replace the pixels of sources that are not the central source
        repl_mask = np.logical_and(segmap > 0, segmap != img_id)
    
        for i in xrange(img.shape[0]):
            for j in xrange(img.shape[1]): 
                if repl_mask[i,j]:
                    # the h and j bands appear to have more spatially correlated
                    # noise so we use the tiny tim convoluted noise here
                    # the v and z bands however have noise that is less spatially
                    # correlated so we'll just a random draw from the existing 
                    # noise to replace those pixels
                    if band in ('h','j'):                  
                        img[i,j] = tt_img[i,j]
                    else:
                        img[i,j] = noise[randint(len_noise)]

    return img    