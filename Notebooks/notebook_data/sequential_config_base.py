# Configuration for training of CNN for STRIDES30 test

import numpy as np
from scipy.stats import norm, uniform, truncnorm, randint
import paltas.Sampling.distributions as dist
from paltas.MainDeflector.simple_deflectors import PEMDShear
from paltas.Sources.sersic import SingleSersicSource
from paltas.PointSource.single_point_source import SinglePointSource

# SEQUENTIAL PROPOSAL INFORMATION
# theta_E, gamma1, gamma2, gamma_lems, e1, e2, x_lens, y_lens, x_src, y_src
# will have to make a joint sampling object for all the params...
seq_mus = [ 0.567,  0.027,  0.167,  1.911,  0.173,  0.037, -0.018,  0.034, -0.035,  0.011,  0.728]
seq_sigmas = [0.023, 0.039, 0.054, 0.11 , 0.093, 0.121, 0.009, 0.008, 0.007, 0.007, 0.284]

# calculated using .fits header
output_ab_zeropoint = 25.1152

kwargs_numerics = {'supersampling_factor':1}

# size of cutout
numpix = 80

# quads_only
#doubles_quads_only = Trues
# point source magnification cut
#ps_magnification_cut = 2

# load in a PSF kernel
from astropy.io import fits
from lenstronomy.Util import kernel_util

#psf_fits_file = '/mnt/home/swagner/Documents/sydney/paltas/datasets/hst_psf/STDPBF_WFC3UV_F814W.fits'
psf_fits_file = '/home/users/sydney3/paltas/datasets/hst_psf/STDPBF_WFC3UV_F814W.fits'

# load in focus diverse PSF maps
with fits.open(psf_fits_file) as hdu:
    psf_kernels = hdu[0].data
psf_kernels = psf_kernels.reshape(-1,101,101)
psf_kernels[psf_kernels<0] = 0

# normalize psf_kernels to sum to 1
psf_sums = np.sum(psf_kernels,axis=(1,2))
psf_sums = psf_sums.reshape(-1,1,1)
normalized_psfs = psf_kernels/psf_sums

# pick random weights to create PSF
def draw_psf_kernel():
	weights = np.random.uniform(size=np.shape(normalized_psfs)[0])
	weights /= np.sum(weights)
	weighted_sum = np.sum(weights.reshape(len(weights),1,1) * normalized_psfs,axis=0)
	return kernel_util.degrade_kernel(weighted_sum,4)

config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'z_lens': truncnorm(-2.5,np.inf,loc=0.5,scale=0.2).rvs,
			'gamma': norm(loc=seq_mus[3],scale=seq_sigmas[3]).rvs,
            # switch to a more uniform Einstein Radius dist.
            'theta_E': norm(loc=seq_mus[0],scale=seq_sigmas[0]).rvs,
            'e1':norm(loc=seq_mus[4],scale=seq_sigmas[4]).rvs,
            'e2':norm(loc=seq_mus[5],scale=seq_sigmas[5]).rvs,
			'center_x':None,
			'center_y':None,
			'gamma1':norm(loc=seq_mus[1],scale=seq_sigmas[1]).rvs,
            'gamma2':norm(loc=seq_mus[2],scale=seq_sigmas[2]).rvs,
			'ra_0':0.0,
			'dec_0':0.0,
		}
	},
    'source':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':truncnorm(-5,np.inf,loc=2.,scale=0.4).rvs,
            # range: 20 to 27, centered at 23.5
            'mag_app':truncnorm(-3./2.,3./2.,loc=23.5,scale=7./3.).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':truncnorm(-(5./5.),np.inf,loc=0.5,scale=0.5).rvs,
			'n_sersic':truncnorm(-1.25,np.inf,loc=3.,scale=1.).rvs,
			'e1':truncnorm(-2.5,2.5,loc=0,scale=0.2).rvs,
            'e2':truncnorm(-2.5,2.5,loc=0,scale=0.2).rvs,
            # see cross_object below
			'center_x':None,
			'center_y':None}

	},
    'lens_light':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':None,
            # range: 17 to 23
            'mag_app':truncnorm(-3./2.,3./2.,loc=20,scale=2.).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':truncnorm(-(1./.8),np.inf,loc=1.0,scale=0.8).rvs,
			'n_sersic':truncnorm(-1.25,np.inf,loc=3.,scale=2.).rvs,
			'e1':truncnorm(-2.5,2.5,loc=0,scale=0.2).rvs,
            'e2':truncnorm(-2.5,2.5,loc=0,scale=0.2).rvs,
            # see cross_object below
			'center_x':None,
			'center_y':None}
	},
    'point_source':{
		'class': SinglePointSource,
		'parameters':{
            # see cross_object below for z,x,y
            'z_point_source':None,
			'x_point_source':None,
			'y_point_source':None,
            # range: 19 to 25
            'mag_app':truncnorm(-3./2.,3./2.,loc=22.,scale=2.).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'mag_pert': dist.MultipleValues(dist=truncnorm(-1/0.3,np.inf,1,0.3).rvs,num=10),
            'compute_time_delays':False
		}
	},
    'cosmology':{
		'parameters':{
			'cosmology_name': 'planck18'
		}
	},
    'psf':{
		'parameters':{
			'psf_type':'PIXEL',
			'kernel_point_source':draw_psf_kernel,
			'point_source_supersampling_factor':1
		}
	},
	'detector':{
		'parameters':{
			'pixel_scale':0.04,'ccd_gain':1.5,'read_noise':3.0,
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':1400.,'sky_brightness':21.9,
			'num_exposures':1,'background_noise':None
		}
	},
	'drizzle':{
		'parameters':{
        		'supersample_pixel_scale':0.040,'output_pixel_scale':0.040,
        		'wcs_distortion':None,
        		'offset_pattern':[(0,0),(0.5,0.5)],
        		'psf_supersample_factor':1
		}
	},
    'cross_object':{
		'parameters':{
            ('main_deflector:center_x,lens_light:center_x'):dist.DuplicateScatter(
                dist=norm(loc=seq_mus[6],scale=seq_sigmas[6]).rvs,scatter=0.005),
            ('main_deflector:center_y,lens_light:center_y'):dist.DuplicateScatter(
                dist=norm(loc=seq_mus[7],scale=seq_sigmas[7]).rvs,scatter=0.005),
            ('source:center_x,source:center_y,point_source:x_point_source,'+
                'point_source:y_point_source'):dist.DuplicateXY(
                x_dist=norm(loc=seq_mus[8],scale=seq_sigmas[8]).rvs,
                y_dist=norm(loc=seq_mus[9],scale=seq_sigmas[9]).rvs),
			('main_deflector:z_lens,lens_light:z_source,source:z_source,'+
                 'point_source:z_point_source'):dist.RedshiftsPointSource(
				z_lens_min=0,z_lens_mean=0.5,z_lens_std=0.2,
				z_source_min=0,z_source_mean=2,z_source_std=0.4)
		}
	}
}