# -*- coding: utf-8 -*-
"""
This script fetches the raw data and labels from their ftp locations
then does the preprocessing neccessary for datahelper.py
"""
# python libs
import os
import re
import tarfile
import gzip
from ftplib import FTP
import time
from shutil import rmtree

# third party libs
from astropy.io import fits
import numpy as np

# local lib
import segmap_helper as s_hlpr
import image_helper as i_hlpr
import table_helper as t_hlpr

data_ftp_url = 'ftp.noao.edu'
data_ftp_dir = 'pub/jeyhan/candels/gds2/'
labels_ftp_url = 'cdsarc.u-strasbg.fr'
labels_ftp_dir = 'pub/cats/J/ApJS/221/11'

# Helpers ---------------------------------------------------------------------
def _get_ftp(url, location, path=None):
    ftp = FTP(url)
    ftp.login()
    
    if path:
        ftp.cwd(path)

    for f in ftp.nlst():
        print 'Fetching {}...'.format(f)
        f_dir = os.path.join(location,f)
        try:
            ftp.retrbinary('RETR {}'.format(f), open(f_dir, 'wb').write)
        except Exception:
            print 'Failed to get {}'.format(f)
    
    ftp.quit()

def _unpack_dir(data_dir, remove=True):
    for f in os.listdir(data_dir):
        f_dir = os.path.join(data_dir, f)
        print 'Unpacking {}...'.format(f)
        tarfile.open(f_dir).extractall(data_dir)
        
        if remove:
            print 'Removing {}'.format(f)
            os.remove(f_dir)
    
    
def _distinct_sources(data_dir):
    sources = {}

    for f in os.listdir(data_dir):
        new_dir = os.path.join(data_dir,f)
        if os.path.isdir(new_dir):
            sources.update(_distinct_sources(new_dir))
        else:
            match = re.search('[a-z]+\d_\d+_', f)
            if match and match.group(0)[:-1] not in sources:
                sources[match.group(0)[:-1]] = data_dir

    return sources

def _new_dir(root, new):
    new_dir = os.path.join(root, new)
    if new not in os.listdir(root):
        os.mkdir(new_dir)
        
    return new_dir

def _simple_log(val):
    with open('./log', 'a') as f:
        f.write(val + '\n')

# if an image is larger than 84x84 we need to see if we can crop it down
def _fits_in_square(seg, src_id):
    dim1, dim2 = np.shape(seg)    
    
    
    min_1, max_1 = (dim1+1,-1)
    min_2, max_2 = (dim2+1,-1)
    
    for d1 in range(dim1):
        for d2 in range(dim2):
            if seg[d1,d2] == src_id:
                if d1 < min_1:
                    min_1 = d1
                elif d1 > max_1:
                    max_1 = d1
                
                if d2 < min_2:
                    min_2 = d2
                elif d2 > max_2:
                    max_2 = d2
                    
    diff_1 = max_1 - min_1
    diff_2 = max_2 - min_2
    # the image is too big too fit within our constraints
    if diff_1 > 83 or diff_2 > 83:
        return None
    # pad the area to make it 84x84
    else:
        def parse_diffs(diff, mn, mx):
            if diff % 2 == 0:
                pad = diff / 2
    
                mx_too_big = mx + pad > 83
                mn_too_small = mn - pad < 0
    
                # we need to know which order to adjust the pad in
    
                if mn_too_small:
                   pad += (pad - mn) * (-1)
                   mn = 0
                   
                   mx += pad
                   
                elif mx_too_big:
                   pad += (pad + mx) - 83
                   mx = 83
                   
                   mn -= pad
                
                else:
                    mn -= pad
                    mx += pad
                
            else:
                pad_1, pad_2 = (diff / 2, diff / 2 + 1)
                
                mx_too_big = mx + pad_2 > 83
                mn_too_small = mn - pad_1 < 0
                
                if mn_too_small:
                    pad_2 += (pad_1 - mn) * (-1)
                    mn = 0
                    
                    mx += pad_2
                elif mx_too_big:
                    pad_1 += (pad_2 + mx) - 83
                    mx = 83
                    
                    mn -= pad_1
                else:
                    mn -= pad_1
                    mx += pad_2
            
            return (mn, mx)
            
        min_1, max_1 = parse_diffs(diff_1, min_1, max_1)
        min_2, max_2 = parse_diffs(diff_2, min_2, max_2)
        
        
            
        
        return (min_1, max_1, min_2, max_2)
        
    
# Helpers ---------------------------------------------------------------------


# START
start = time.time()
# DATA ------------------------------------------------------------------------

# this dir should always exist until I find a way to get TinyTim data
data_dir = _new_dir('../','data')

# tmp dir for the raw files
tmp_dir = _new_dir(data_dir, 'tmp')

if len(os.listdir(tmp_dir)) == 0:
    # put all the raw data in the tmp directory
    _get_ftp(data_ftp_url, tmp_dir, path=data_ftp_dir)
    
    # unpack all the tarballs
    _unpack_dir(tmp_dir)

# get the distinct image names
sources =  _distinct_sources(tmp_dir)

# Apply preprocessing to the images and save them to imgs 
f_format = 'GDS_{}_{}.fits'
bands = ['h','j','v','z']

tt_imgs = {b:fits.getdata('../data/tinytim/{}.fits'.format(b)) for b in bands}

imgs_dir = os.path.join(data_dir, 'imgs')    
if 'imgs' not in os.listdir(data_dir):
    os.mkdir(imgs_dir)    

img_count = 1
img_total = len(sources)
for s in sources.iterkeys():
    print 'Working on source {} of {}: {}'.format(img_count, img_total, s)
    img_count += 1
    
    s_dir = sources[s]
    img_id = int(s[(s.index('_')+1):])

    # fix segmap
    seg_dir = os.path.join(s_dir, f_format.format(s, 'segmap'))
    segmap = fits.getdata(seg_dir)

    valid_img = True
    pad = False
    crop = False
    crop_help = None
    pad_help = None    
    
    # if our image is not 84x84
    dim = np.shape(segmap)[0]
    if dim != 84:
        # we need pad the edges to pad the with the correct noise will 
        # add '1' pixels to emulate a non central source
        if dim < 84:
            pad = True

            pad_help = 84 - dim

            d1_pad = np.ones((pad_help,dim))
            d2_pad = np.ones((84,pad_help))
            
            segmap = np.append(segmap, d1_pad)
            segmap = np.append(segmap, d2_pad)
            
            
            
        else:
            crop_help = _fits_in_square(segmap,img_id)
            if crop_help:
                crop = True
                
                d1_mn, d1_mx, d2_mn, d2_mx = crop_help                
                segmap = segmap[d1_mn:d1_mx, d2_mn:d2_mx]
            else:
                valid_img = False
    
    if not valid_img:
        err = '{} skipped: dimensions are too large and the segmap will not fit '
        _simple_log(err.format(s))
        continue

    imgs = {}
    try:
        for b in bands:
            i_dir = os.path.join(s_dir, f_format.format(s, b))
            raw_img = fits.getdata(i_dir)                        
            
            if crop:
                _simple_log('cropping {} band {}'.format(s,b))
                d1_mn, d1_mx, d2_mn, d2_mx = crop_help
                # crop image 
                raw_img = raw_img[d1_mn:d1_mx, d2_mn:d2_mx]                
            elif pad:
                # pad with 0's it doesn't matter what the value is, as the 
                # img transformer will replace it with random noise
                _simple_log('padding {} band {}'.format(s, b))                
                
                d1_pad = np.ones((pad_help,dim))
                d2_pad = np.ones((84,pad_help))
                
                raw_img = np.append(d1_pad)
                raw_img = np.append(d2_pad)

            imgs[b] = raw_img
    except Exception:
        err = '{} not included because all filters could not be loaded'
        _simple_log(err.format(s))
        continue

    segmap = s_hlpr.transform_segmap(imgs.values(), segmap)    

    # fix images in each band
    for b in bands:        
        imgs[b] = i_hlpr.transform_image(imgs[b], b, segmap, tt_imgs[b])   

    # combine into one image and save
    cmb_img = np.array(imgs.values())

    cmb_dir = os.path.join(imgs_dir, '{}.fits'.format(s))
    fits.PrimaryHDU(cmb_img).writeto(cmb_dir)

#def process_source(kvp):
#    f_format = 'GDS_{}_{}.fits'
#    bands = ['h','j','v','z']
#    s, s_dir = kvp
#    
#    seg_dir = os.path.join(s_dir, f_format.format(s, 'segmap'))
#    segmap = fits.getdata(seg_dir)
#    
#    imgs = {}
#    
#    err = False
#    msg = ''
#    try:
#        for b in bands:
#            i_dir = os.path.join(s_dir, f_format.format(s, b))
#            raw_img = fits.getdata(i_dir)  
#
#            
#    
#            imgs[b] = raw_img
#    except Exception as e:
#        err = True
#        if e.args
#        msg = '{} not included because all filters could not be loaded'.format(s)
#        
#        

# clean up
rmtree(tmp_dir, ignore_errors=True)
# DATA ------------------------------------------------------------------------

# LABELS ----------------------------------------------------------------------
lbl_dir = os.path.join(data_dir, 'labels')
if 'labels' not in os.listdir(data_dir):
    os.mkdir(lbl_dir)

# put all the raw data in the tmp directory
_get_ftp(labels_ftp_url, lbl_dir, path=labels_ftp_dir)

# keep the files we want to keep
lbl_tables = ['table2.dat.gz', 'table3.dat.gz']
read_me = ['ReadMe']
files_to_keep = lbl_tables + read_me

for f in os.listdir(lbl_dir):
    f_dir = os.path.join(lbl_dir, f)
    
    if f not in files_to_keep:
        print 'Removing {}'.format(f)
        os.remove(f_dir)
    elif f in lbl_tables:
        print 'Extracting {}...'.format(f)
        with gzip.open(f_dir, 'r') as gz, open(f_dir[:-3], 'w') as f:
            f.write(gz.read())

        # clean up
        print 'Removing {}'.format(f)
        os.remove(f_dir)
        
# make our labels table
tbl_2 = lbl_tables[0]
tbl_2_path = os.path.join(lbl_dir, tbl_2[:-3])
new_tbl_2 = os.path.join(lbl_dir, 'labels.csv')
t_hlpr.extract_from_table2(tbl_2_path, new_tbl_2)
# LABELS ----------------------------------------------------------------------

#END
# TODO format into minutes and hours
print 'Process finished in {} seconds'.format(time.time() - start)
