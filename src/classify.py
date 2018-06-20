import argparse
import os

import numpy as np
from astropy.io import fits

# Mean and Variance calculations from:
# http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.214.8508&rep=rep1&type=pdf


def classify_img(*ignore, h='h.fits', j='j.fits', v='v.fits', z='z.fits'):
    naxis1, naxis2 = _validate_args(ignore, h, j, v, z)

    # create placeholder fits files




def _validate_args(ignore, h, j, v, z):
    if ignore:
        raise Exception('Positional arguments are not allowed, please use named arguments')

    # validate files names
    for fname in [h,j,v,z]:
        if not os.path.exists(fname) or not os.path.isfile(fname):
            raise ValueError('Invalid file {}'.format(fname))

    # validate file sizes
    naxis1, naxis2 = None, None
    for fname in [h,j,v,z]:
        hdul = fits.open(fname)
        hdu = hdul[0]
        if naxis1 is None:
            naxis1 = hdu.header['NAXIS1']
            naxis2 = hdu.header['NAXIS2']
        else:
            if naxis1 != hdu.header['NAXIS1']:
                msg = 'Images are not same dims NAXIS1: {} != {}'
                msg = msg.format(naxis1, hdu.header['NAXIS1'])
                raise Exception(msg)
            elif naxis2 != hdu.header['NAXIS2']:
                msg = 'Images are not same dims NAXIS2: {} != {}'
                msg = msg.format(naxis1, hdu.header['NAXIS2'])
                raise Exception(msg)

        hdul.close()
    return naxis1, naxis2

# http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf, eq. 4
def _iterative_mean(n, prev_mean, x_n):
    return prev_mean + (x_n - prev_mean)/n

# http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf, eq. 24
def _iterative_variance(n, prev_var, x_n, prev_mean, curr_mean, unbiased_est):
    var = prev_var + (x_n - prev_mean) * (x_n - curr_mean)

    if unbiased_est:
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.214.8508&rep=rep1&type=pdf, eq. II.2 (right after)
        return 1/(n-1) * var
    else:
        return var


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('h', help='fits file containing img in H band')
    parser.add_argument('j', help='fits file containing img in J band')
    parser.add_argument('v', help='fits file containing img in V band')
    parser.add_argument('z', help='fits file containing img in Z band')

    args = parser.parse_args()
    classify_img(h=args.h, j=args.j, v=args.v, z=args.z)
