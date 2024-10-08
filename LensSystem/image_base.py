"""
Base class for images
"""
import numpy as np
from scipy.ndimage import rotate
from astropy.visualization import simple_norm

class ImageBase():

    def __init__(self):
        self.image_data = None

    def center_crop(self,data,numpix):
        if numpix > np.shape(data)[0]:
            # pad with zeros
            new_data = np.zeros([numpix,numpix])
            offset = int(np.shape(data)[0]/2)
            x_targ,y_targ = int(numpix/2),int(numpix/2)
            new_data[x_targ-offset:x_targ+offset,
            y_targ-offset:y_targ+offset] = data
        else:
            offset = int(numpix/2)
            x_targ,y_targ = int(np.shape(data)[0]/2),int(np.shape(data)[1]/2)
            new_data = data[x_targ-offset:x_targ+offset,
            y_targ-offset:y_targ+offset]

        return new_data

    def retrieve_image(self,rotation_angle=None,flip=False,numpix=None):
        """
        Args: 
            rotate (float): Optional, amount to rotate image in degrees
            flip (boolean): default=False, if True, mirrors image left/right

        Returns:
            data (array): Array of image data in counts per second
        """

        data = self.image_data

        if numpix is not None:
            data = self.center_crop(data,numpix)

        if rotation_angle is not None:
            data = rotate(data,rotation_angle)
        if flip:
            data = np.fliplr(data)

        return data

    def image_norm(self,type='log',min_cutoff=1e-6,numpix=None):
        """
        Args: 
            type (string): default='log', name of function used to create 
                normalization. 
            min_cutoff (float): default=1e-6, minimum value of the 
                normalization scale
        """
        if numpix is None:
            norm = simple_norm(self.image_data,stretch=type,
            min_cut=min_cutoff)
        else:
           norm = simple_norm(self.center_crop(self.image_data,numpix),stretch=type,
            min_cut=min_cutoff) 
        return norm

    def image_positions(self):
        print("Not Implemented")
        return