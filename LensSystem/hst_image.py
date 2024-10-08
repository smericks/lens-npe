"""
Class to store HST image, initialized with a .fits file
"""
from LensSystem.image_base import ImageBase
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import numpy as np
from visualization_utils import header_ab_zeropoint


class HSTImage(ImageBase):
    """
    Args: 
        fits_file_path (string): Path to .fits file
        lens_catalog_row (pandas Series): Single row of lens system from 
            lens catalog
    """


    def __init__(self,fits_file_path,lens_catalog_row):

        self.fits_file_path = fits_file_path

        # load in image from .fits file
        with fits.open(self.fits_file_path) as hdu:
            
            # read in data 
            data = hdu[0].data

            # reflect dec axis for correct E/W convention
            data = np.flip(data,axis=0)
            wcs = WCS(hdu[0].header)
            ra_targ = hdu[0].header['RA_TARG']
            dec_targ = hdu[0].header['DEC_TARG']

            # apply any needed adjustments to center of lens system
            ra_targ += lens_catalog_row['ra_adjust']
            dec_targ += lens_catalog_row['dec_adjust']

            # retrieve array coordinates of lens center
            x_targ,y_targ = wcs.world_to_array_index(SkyCoord(ra=ra_targ,
                    dec=dec_targ,unit="deg"))
            
            # transform x_targ to reflected dec coordinates
            num_x_pixels = hdu[0].header['NAXIS2']
            x_targ = int(x_targ - 2*(x_targ - (num_x_pixels/2)))

            # find out # arseconds per pixel
            self.arcseconds_per_pixel = hdu[0].header['D001SCAL']

        # retreive offset from catalog and crop data
        offset = int(lens_catalog_row['cutout_size']/2)
        cropped_data = data[x_targ-offset:x_targ+offset,
        y_targ-offset:y_targ+offset]

        # save cropped data as image
        self.image_data = cropped_data

    def ab_zeropoint(self):
        """
        calculates AB zeropoint using information from fits header
    
        Returns:
            (float): AB zeropoint
        """
        with fits.open(self.fits_file_path) as hdu:
            return header_ab_zeropoint(hdu)

