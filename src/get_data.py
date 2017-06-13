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
import fcntl

from multiprocessing import Pool
from itertools import repeat

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

RUN_PARALLEL = True

# Helpers ---------------------------------------------------------------------
def _get_ftp(url, location, path=None):
    ftp = FTP(url)
    ftp.login()
    
    if path:
        ftp.cwd(path)

    for f in ftp.nlst():
        print('Fetching {}...'.format(f))
        f_dir = os.path.join(location,f)
        try:
            ftp.retrbinary('RETR {}'.format(f), open(f_dir, 'wb').write)
        except Exception:
            print('Failed to get {}'.format(f))
    
    ftp.quit()

def _unpack_dir(data_dir, remove=True):
    for f in os.listdir(data_dir):
        f_dir = os.path.join(data_dir, f)
        print('Unpacking {}...'.format(f))
        tarfile.open(f_dir).extractall(data_dir)
        
        if remove:
            print('Removing {}'.format(f))
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
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(val + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)

def _simple_log_lines(vals):
    with open('./log', 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for val in vals:        
            f.write(val + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)

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
    if diff_1 > 84 or diff_2 > 84:
        return None
    # pad the area to make it 84x84
    else:
        def parse_diffs(dim, diff, mn, mx):                        
            mx_val = dim-1            
            
            if diff % 2 == 0:
                pad = (84-diff) // 2
    
                mx_too_big = mx + pad > mx_val
                mn_too_small = mn - pad < 0
    
                # we need to know which order to adjust the pad in
    
                if mn_too_small:
                   pad += abs(pad - mn)
                   mn = 0
                   
                   mx += pad
                   
                elif mx_too_big:
                   pad += (pad + mx) - mx_val
                   mx = mx_val
                   
                   mn -= pad
                
                else:
                    mn -= pad
                    mx += pad
                
            else:
                pad_1, pad_2 = (((84-diff) // 2), (((84-diff) // 2) + 1))
                
                mn_too_small = mn - pad_1 < 0
                mx_too_big = mx + pad_2 > mx_val
                
                if mn_too_small:
                    pad_2 += abs(pad_1 - mn)
                    mn = 0
                    
                    mx += pad_2
                elif mx_too_big:
                    pad_1 += (pad_2 + mx) - mx_val
                    mx = mx_val
                    
                    mn -= pad_1
                else:
                    mn -= pad_1
                    mx += pad_2
            
            return (mn, mx)
            
        min_1, max_1 = parse_diffs(dim1, diff_1, min_1, max_1)
        min_2, max_2 = parse_diffs(dim2, diff_2, min_2, max_2)  
    
        return (min_1, max_1, min_2, max_2)
    
def _pad_dim(img, axs):
    dims = list(np.shape(img))
    dim = dims[axs]
    
    diff = 84 - dim
    
    if diff % 2 == 0:
        dims[axs] = diff // 2
        
        pad = np.ones(tuple(dims))
        img = np.append(pad, img, axis=axs)
        img = np.append(img, pad, axis=axs)
    else:
       dims[axs] = diff // 2
       
       pad = np.ones(tuple(dims))
       img = np.append(pad, img, axis=axs)
       
       dims[axs] = diff // 2 + 1
       
       pad = np.ones(tuple(dims))
       img = np.append(img, pad, axis=axs)

    return img
    
def process_image(args):
    kvp, tinytims = args
    s, s_dir = kvp
    print(s)
    
    img_id = int(s[(s.index('_')+1):])    
    
    to_dir = '../data/imgs'        
    f_format = 'GDS_{}_{}.fits'
    bands = ['h','j','v','z']

    # log messages
    # TODO use an acutal logging library    
    msgs = []
    m_pad = '{} {} needs pad'
    m_crop = '{} {} needs crop'
    m_skip = '{} skipped: dimensions are too large and the segmap will not fit'
    m_shp =  'from {} new shape {}'  
    m_filt = '{} not included because all filters could not be loaded'


    # SEGMAP ------------------------------------------------------------------
    seg_dir = os.path.join(s_dir, f_format.format(s, 'segmap'))
    segmap = fits.getdata(seg_dir)

    crop_help_d1 = None 
    crop_help_d2 = None
    edited = False    

    dim1, dim2 = np.shape(segmap)  
    
    # if the dimension is smaller than 84 we need to pad the edges up to 84
    if dim1 < 84:               
        msgs.append(m_pad.format(s, 'dim1'))
        edited = True
        segmap = _pad_dim(segmap, 0)           
    
    # if the dimension is larger than 84 we need to crop the image down to 84
    elif dim1 > 84:
        msgs.append(m_crop.format(s, 'dim1'))
        edited = True
        crop_help_d1 = _fits_in_square(segmap, img_id)
        
        # it possible that only one dimension needs to be cropped to only adjust
        # the dimnsion we are dealing with
        if crop_help_d1:
            mn, mx, _, _ = crop_help_d1
            segmap = segmap[mn:mx, :]
        else:
             msgs.append(m_skip.format(s))
             _simple_log_lines(msgs)
             return
             
    # reset our variables just in case dim1 was changed
    dim1, dim2 = np.shape(segmap)        
    
    if dim2 < 84:
        msgs.append(m_pad.format(s, 'dim1'))
        edited = True
        segmap = _pad_dim(segmap, 1)
    elif dim2 > 84:
        msgs.append(m_crop.format(s, 'dim1'))
        edited = True
        crop_help_d2 = _fits_in_square(segmap, img_id)           
        
        if crop_help_d2:
            _, _, mn, mx = crop_help_d2
            segmap = segmap[:,mn:mx]
        else:
            msgs.append(m_skip.format(s))
            _simple_log_lines(msgs)
            return
            
            
    if edited:
        msgs.append(m_shp.format((dim1,dim2), np.shape(segmap)))
    # SEGMAP ------------------------------------------------------------------
    
    # IMAGES ------------------------------------------------------------------
    imgs = {}
    try:
        for b in bands:
            i_dir = os.path.join(s_dir, f_format.format(s, b))
            raw_img = fits.getdata(i_dir)                        
            
            dim1, dim2 = np.shape(raw_img)

            if dim1 < 84:
                raw_img = _pad_dim(raw_img, 0)
            elif dim1 > 84:
                mn, mx, _, _ = crop_help_d1
                raw_img = raw_img[mn:mx, :]
            
            dim1, dim2 = np.shape(raw_img)                                
            
            if dim2 < 84:
                raw_img = _pad_dim(raw_img, 1)
            elif dim2 > 84:
                _, _, mn, mx = crop_help_d2
                raw_img = raw_img[:, mn:mx]

            imgs[b] = raw_img
    except Exception:
        msgs.append(m_filt.format(s))
        _simple_log_lines(msgs)
        return
    # IMAGES ------------------------------------------------------------------

    # if we made any changes to the images write them to the log
    if len(msgs) > 0:
        _simple_log_lines(msgs)

    # PREPROCESSING -----------------------------------------------------------
    segmap = s_hlpr.transform_segmap(imgs.values(), segmap)    

    # fix images in each band
    for b in bands:        
        imgs[b] = i_hlpr.transform_image(imgs[b], img_id, s, b, segmap, tt_imgs[b])   

    # PREPROCESSING -----------------------------------------------------------

    # combine into one image and save
    cmb_img = np.dstack(tuple([imgs[b] for b in bands]))

    cmb_dir = os.path.join(to_dir, '{}.fits'.format(s))
    fits.PrimaryHDU(cmb_img).writeto(cmb_dir)
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
bands = ['h','j','v','z']
tt_imgs = {b:fits.getdata('../data/tinytim/{}.fits'.format(b)) for b in bands}

imgs_dir = os.path.join(data_dir, 'imgs')    
if 'imgs' not in os.listdir(data_dir):
    os.mkdir(imgs_dir)    

if RUN_PARALLEL:
    print('Asyncing!')
    Pool().map(process_image, zip(sources.items(), repeat(tt_imgs)))
else:
    print('Sequentialing!')
    for args in zip(sources.items(), repeat(tt_imgs)):
        process_image(args)

# clean up
#rmtree(tmp_dir, ignore_errors=True)
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
        print('Removing {}'.format(f))
        os.remove(f_dir)
    elif f in lbl_tables:
        print('Extracting {}...'.format(f))
        with gzip.open(f_dir, 'r') as gz, open(f_dir[:-3], 'w') as f:
            f.write(gz.read())

        # clean up
        print('Removing {}'.format(f))
        os.remove(f_dir)
        
# make our labels table
tbl_2 = lbl_tables[0]
tbl_2_path = os.path.join(lbl_dir, tbl_2[:-3])
new_tbl_2 = os.path.join(lbl_dir, 'labels.csv')
t_hlpr.extract_from_table2(tbl_2_path, new_tbl_2)
# LABELS ----------------------------------------------------------------------

#END
# TODO format into minutes and hours
print('Process finished in {} seconds'.format(time.time() - start))
