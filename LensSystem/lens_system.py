"""
Class to store lens image and lens models
"""

from LensSystem.hst_image import HSTImage
from LensSystem.forward_model_image import ForwardModelImage
from LensSystem.paltas_model_image import PaltasModelImage
import pandas as pd
from LensSystem.doppelganger_utils import *
from astropy.io import fits
import visualization_utils
from lenstronomy.Util.data_util import cps2magnitude

class LensSystem():
    """
    Required Args: 
        name (string): Name of lensing system, matches name in lens catalog
        catalog_path (string): Path to .csv containing lens catalog information
    Optional Args: 
        hst_file (string): .fits file containing drizzled HST image 
        forward_model_file (string): .txt file containing best fit parameters 
            from forward modeling
        paltas_config (string): .py file containing paltas configuration 
    """

    def __init__(self,name,catalog_path,hst_file=None,forward_model_file=None,paltas_config=None,make_image=True):

        # retrieve row of lens information from catalog .csv as a Pandas series
        catalog_df = pd.read_csv(catalog_path)
        self.lens_catalog_row = catalog_df[catalog_df['name'] == name].iloc[0]

        self.hst_image=None
        self.forward_model=None
        self.paltas_model=None

        if hst_file is not None:
            self.hst_image = HSTImage(hst_file,self.lens_catalog_row)
        
        if forward_model_file is not None:
            self.forward_model = ForwardModelImage(forward_model_file,
                make_image=make_image)

        if paltas_config is not None:
            self.paltas_model = PaltasModelImage(paltas_config)
            
    
    def _write_config_imports(self,out_file):
        """
        Writes import statements to file object out_file
        
        Args:
            out_file: opened file to write to
        """
        lines = ['import numpy as np',
            'import paltas.Sampling.distributions as dist',
            'from paltas.MainDeflector.simple_deflectors import PEMDShear',
            'from paltas.Sources.sersic import SingleSersicSource',
            'from paltas.PointSource.single_point_source import SinglePointSource',
            'import pickle',
            'from paltas.Utils.cosmology_utils import get_cosmology, apparent_to_absolute']
        out_file.write('\n'.join(lines))

    def _write_config_setup(self,out_file,no_noise=False):
        """
        Writes block of text with initial calculations to out_file

        Args:
            out_file: opened file to write to
        """
        # no_noise option
        if no_noise:
            out_file.write('no_noise = True')
            out_file.write('\n\n')
        # AB zeropoint
        out_file.write('output_ab_zeropoint = %f'%(self.hst_image.ab_zeropoint()))
        # numerics kwargs
        out_file.write('\n\n')
        out_file.write('kwargs_numerics = {\u0027supersampling_factor\u0027:1}')
        # numpix
        out_file.write('\n\n')
        out_file.write('numpix = %d'%(self.lens_catalog_row['cutout_size']))
        # caustic area
        out_file.write('\n\n')
        out_file.write('compute_caustic_area = True')
        # psf map
        out_file.write('\n\n')
        out_file.write('# load PSF map from lenstronomy fitting results\n')
        out_file.write('f = open(\u0027' + self.forward_model.results_file + 
            '\u0027,\u0027rb\u0027)\n')
        out_file.write('multi_band_list,_,_,_ = pickle.load(f)\n')
        out_file.write('psf_map = multi_band_list[0][1][\u0027kernel_point_source\u0027]')
        # cosmology
        out_file.write('\n\n')
        out_file.write('cosmology_params = { \u0027cosmology_name\u0027: \u0027planck18\u0027 }\n')
        out_file.write('cosmo = get_cosmology(cosmology_params)')

    def _write_emp_PSF_setup(self,out_file):
        out_file.write('\n\n')
        out_file.write('from astropy.io import fits\n')
        out_file.write('from lenstronomy.Util import kernel_util\n')
        out_file.write('psf_fits_file = '+
            '\u0027/Users/smericks/Desktop/StrongLensing/paltas/datasets/'+
            'hst_psf/STDPBF_WFC3UV_F814W.fits\u0027\n')
        out_file.write('with fits.open(psf_fits_file) as hdu:\n'+'\tpsf_kernels = hdu[0].data\n'+
            'psf_kernels = psf_kernels.reshape(-1,101,101)\n'+'psf_kernels[psf_kernels<0] = 0\n\n'+
            'psf_sums = np.sum(psf_kernels,axis=(1,2))\n'+'psf_sums = psf_sums.reshape(-1,1,1)\n'+
            'normalized_psfs = psf_kernels/psf_sums\n'+'def draw_psf_kernel():\n'+
            '\tweights = np.random.uniform(size=np.shape(normalized_psfs)[0])\n'+
            '\tweights = np.random.uniform(size=np.shape(normalized_psfs)[0])\n'+
            '\tweights /= np.sum(weights)\n'+
            '\tweighted_sum = np.sum(weights.reshape(len(weights),1,1) * normalized_psfs,axis=0)\n'+
            '\treturn kernel_util.degrade_kernel(weighted_sum,4)\n\n')

    def _write_main_deflector(self,out_file):
        """
        Writes block of text with main_deflector dict for paltas config

        Args:
            out_file: opened file to write to
        """
        out_file.write('\n')
        out_file.write('\t\u0027main_deflector\u0027:{\n')
        out_file.write('\t\t\u0027class\u0027: PEMDShear,\n')
        out_file.write('\t\t\u0027parameters\u0027:{\n')
        # redshift
        z = self.lens_catalog_row['lens_redshift']
        if np.isnan(z):
            z = 0.5
        out_file.write('\t\t\t\u0027z_lens\u0027:%.3f,\n'%(z))
        # PEMD profile
        kwargs_result = self.forward_model.kwargs_result
        kwargs_PEMD = kwargs_result['kwargs_lens'][0]
        kwargs_shear = kwargs_result['kwargs_lens'][1]
        out_file.write('\t\t\t\u0027gamma\u0027: %.10f,\n'%(kwargs_PEMD['gamma']))
        out_file.write('\t\t\t\u0027theta_E\u0027: %.10f,\n'%(kwargs_PEMD['theta_E']))
        out_file.write('\t\t\t\u0027e1\u0027: %.10f,\n'%(kwargs_PEMD['e1']))
        out_file.write('\t\t\t\u0027e2\u0027: %.10f,\n'%(kwargs_PEMD['e2']))
        # manual adjustment for re-centering of J2145
        if self.lens_catalog_row['name'] == 'J2145+6345':
            out_file.write('\t\t\t\u0027center_x\u0027: %.10f,\n'%(kwargs_PEMD['center_x']+0.16))
            out_file.write('\t\t\t\u0027center_y\u0027: %.10f,\n'%(kwargs_PEMD['center_y']-0.32))
        else:
            out_file.write('\t\t\t\u0027center_x\u0027: %.10f,\n'%(kwargs_PEMD['center_x']))
            out_file.write('\t\t\t\u0027center_y\u0027: %.10f,\n'%(kwargs_PEMD['center_y']))
        # shear
        out_file.write('\t\t\t\u0027gamma1\u0027: %.10f,\n'%(kwargs_shear['gamma1']))
        out_file.write('\t\t\t\u0027gamma2\u0027: %.10f,\n'%(kwargs_shear['gamma2']))
        if self.lens_catalog_row['name'] == 'J2145+6345':
            # TODO: see how bad it is to not change?
            out_file.write('\t\t\t\u0027ra_0\u0027: %.10f,\n'%(kwargs_shear['ra_0']))#+0.16))
            out_file.write('\t\t\t\u0027dec_0\u0027: %.10f\n'%(kwargs_shear['dec_0']))#-0.32))
        else:
            out_file.write('\t\t\t\u0027ra_0\u0027: %.10f,\n'%(kwargs_shear['ra_0']))
            out_file.write('\t\t\t\u0027dec_0\u0027: %.10f\n'%(kwargs_shear['dec_0']))
        # close brackets
        out_file.write('\t\t}\n')
        out_file.write('\t},')
    
    def _write_source(self,out_file):
        """
        Writes block of text with source dict for paltas config

        Args:
            out_file: opened file to write to
        """
        out_file.write('\n')
        out_file.write('\t\u0027source\u0027:{\n')
        out_file.write('\t\t\u0027class\u0027: SingleSersicSource,\n')
        out_file.write('\t\t\u0027parameters\u0027:{\n')
        # redshift
        z = self.lens_catalog_row['source_redshift']
        if np.isnan(z):
            z = 2.0
        out_file.write('\t\t\t\u0027z_source\u0027:%.3f,\n'%(z))
        # sersic profile
        profile_name = self.forward_model.kwargs_model['source_light_model_list'][0]
        kwargs_source = self.forward_model.kwargs_result['kwargs_source'][0]
        # magnitude
        mag = visualization_utils.amp_2_apparent_mag(profile_name,kwargs_source,self.hst_image.ab_zeropoint())
        if np.isnan(mag):
            print("magnitude error! setting to AB zeropoint")
            mag = self.hst_image.ab_zeropoint()
        out_file.write('\t\t\t\u0027mag_app\u0027: %.10f,\n'%(mag))
        out_file.write('\t\t\t\u0027output_ab_zeropoint\u0027: output_ab_zeropoint,\n')
        out_file.write('\t\t\t\u0027R_sersic\u0027: %.10f,\n'%(kwargs_source['R_sersic']))
        out_file.write('\t\t\t\u0027n_sersic\u0027: %.10f,\n'%(kwargs_source['n_sersic']))
        out_file.write('\t\t\t\u0027e1,e2\u0027: dist.EllipticitiesTranslation(q_dist=1,phi_dist=0),\n')
        if self.lens_catalog_row['name'] == 'J2145+6345':
            out_file.write('\t\t\t\u0027center_x\u0027: %.10f,\n'%(kwargs_source['center_x']+0.16))
            out_file.write('\t\t\t\u0027center_y\u0027: %.10f\n'%(kwargs_source['center_y']-0.32))
        else:
            out_file.write('\t\t\t\u0027center_x\u0027: %.10f,\n'%(kwargs_source['center_x']))
            out_file.write('\t\t\t\u0027center_y\u0027: %.10f\n'%(kwargs_source['center_y']))
        # close brackets
        out_file.write('\t\t}\n')
        out_file.write('\t},')

    def _write_lens_light(self,out_file):
        """
        Writes block of text with source dict for paltas config

        Args:
            out_file: opened file to write to
        """
        out_file.write('\n')
        out_file.write('\t\u0027lens_light\u0027:{\n')
        out_file.write('\t\t\u0027class\u0027: SingleSersicSource,\n')
        out_file.write('\t\t\u0027parameters\u0027:{\n')
        # redshift
        z = self.lens_catalog_row['lens_redshift']
        if np.isnan(z):
            z = 0.5
        out_file.write('\t\t\t\u0027z_source\u0027:%.3f,\n'%(z))
        # sersic profile
        profile_name = self.forward_model.kwargs_model['lens_light_model_list'][0]
        kwargs_sersic = self.forward_model.kwargs_result['kwargs_lens_light'][0]
        # magnitude
        mag = visualization_utils.amp_2_apparent_mag(profile_name,kwargs_sersic,self.hst_image.ab_zeropoint())
        if np.isnan(mag):
            print("magnitude error! setting to AB zeropoint")
            mag = self.hst_image.ab_zeropoint()
        out_file.write('\t\t\t\u0027mag_app\u0027: %.10f,\n'%(mag))
        out_file.write('\t\t\t\u0027output_ab_zeropoint\u0027: output_ab_zeropoint,\n')
        out_file.write('\t\t\t\u0027R_sersic\u0027: %.10f,\n'%(kwargs_sersic['R_sersic']))
        out_file.write('\t\t\t\u0027n_sersic\u0027: %.10f,\n'%(kwargs_sersic['n_sersic']))
        out_file.write('\t\t\t\u0027e1\u0027: %.10f,\n'%(kwargs_sersic['e1']))
        out_file.write('\t\t\t\u0027e2\u0027: %.10f,\n'%(kwargs_sersic['e2']))      
        if self.lens_catalog_row['name'] == 'J2145+6345':
            out_file.write('\t\t\t\u0027center_x\u0027: %.10f,\n'%(kwargs_sersic['center_x']+0.16))
            out_file.write('\t\t\t\u0027center_y\u0027: %.10f\n'%(kwargs_sersic['center_y']-0.32))
        else:
            out_file.write('\t\t\t\u0027center_x\u0027: %.10f,\n'%(kwargs_sersic['center_x']))
            out_file.write('\t\t\t\u0027center_y\u0027: %.10f\n'%(kwargs_sersic['center_y']))
        # close brackets
        out_file.write('\t\t}\n')
        out_file.write('\t},')

    def _write_point_source(self,out_file):
        """
        Writes block of text with point source dict for paltas config

        Args:
            out_file: opened file to write to
        """
        out_file.write('\n')
        out_file.write('\t\u0027point_source\u0027:{\n')
        out_file.write('\t\t\u0027class\u0027: SinglePointSource,\n')
        out_file.write('\t\t\u0027parameters\u0027:{\n')
        # redshift
        z = self.lens_catalog_row['source_redshift']
        if np.isnan(z):
            z = 2.0
        out_file.write('\t\t\t\u0027z_point_source\u0027:%.3f,\n'%(z))
        # magnitude
        source_amp, corr = self.forward_model.image_amp_to_source_amp()
        source_mag = visualization_utils.ps_amp_2_apparent_mag(source_amp,
            self.hst_image.ab_zeropoint())
        out_file.write('\t\t\t\u0027mag_app\u0027: %.10f,\n'%(source_mag))
        out_file.write('\t\t\t\u0027output_ab_zeropoint\u0027: output_ab_zeropoint,\n')
        # x,y position
        ra_ps,dec_ps = self.forward_model.image_pos_to_source_pos()
        if self.lens_catalog_row['name'] == 'J2145+6345':
            out_file.write('\t\t\t\u0027x_point_source\u0027: %.10f,\n'%(ra_ps+0.16))
            out_file.write('\t\t\t\u0027y_point_source\u0027: %.10f,\n'%(dec_ps-0.32))
        else:
            out_file.write('\t\t\t\u0027x_point_source\u0027: %.10f,\n'%(ra_ps))
            out_file.write('\t\t\t\u0027y_point_source\u0027: %.10f,\n'%(dec_ps))
        out_file.write('\t\t\t\u0027compute_time_delays\u0027:False,\n')
        # magnification pertubations
        out_file.write('\t\t\t\u0027mag_pert\u0027:' + 
            '[%.3f,%.3f,%.3f,%.3f]\n'%(corr[0],corr[1],corr[2],corr[3]))
        # close brackets
        out_file.write('\t\t}\n')
        out_file.write('\t},')

        # lens equation solver kwargs
        out_file.write('\n')
        out_file.write('\t\u0027lens_equation_solver\u0027:{\n')
        out_file.write('\t\t\u0027parameters\u0027:{\n')
        out_file.write('\t\t\t\u0027search_window\u0027: %.3f,\n'%(
            self.lens_catalog_row['cutout_size']*0.04))
        out_file.write('\t\t\t\u0027min_distance\u0027: %.3f\n'%(0.04))
        # close brackets
        out_file.write('\t\t}\n')
        out_file.write('\t},')

    def _write_PSF_config(self,out_file):
        """
        Writes block of text with PSF dict for paltas config

        Args:
            out_file: opened file to write to
        """
        out_file.write('\n')
        out_file.write('\t\u0027psf\u0027:{\n')
        out_file.write('\t\t\u0027parameters\u0027:{\n')
        # PSF params
        out_file.write('\t\t\t\u0027psf_type\u0027:\u0027PIXEL\u0027,\n')
        out_file.write('\t\t\t\u0027kernel_point_source\u0027:psf_map,\n')
        out_file.write('\t\t\t\u0027point_source_supersampling_factor\u0027:1\n')
        # close brackets
        out_file.write('\t\t}\n')
        out_file.write('\t},')

    def _write_emp_PSF_config(self,out_file):
        out_file.write('\n')
        out_file.write('\t\u0027psf\u0027:{\n')
        out_file.write('\t\t\u0027parameters\u0027:{\n')
        # PSF params
        out_file.write('\t\t\t\u0027psf_type\u0027:\u0027PIXEL\u0027,\n')
        out_file.write('\t\t\t\u0027kernel_point_source\u0027:draw_psf_kernel,\n')
        out_file.write('\t\t\t\u0027point_source_supersampling_factor\u0027:1\n')
        # close brackets
        out_file.write('\t\t}\n')
        out_file.write('\t},')

    def _write_pixel_grid(self,out_file):
        """
        Writes block of text with pixel grid dict for paltas config

        Args:
            out_file: opened file to write to
        """
        out_file.write('\n')
        out_file.write('\t\u0027pixel_grid\u0027:{\n')
        out_file.write('\t\t\u0027parameters\u0027:{\n')
        kwargs_data = self.forward_model.multi_band_list[0][0]
        # need to account for case when desired numpix != fm numpix
        numpix = self.lens_catalog_row['cutout_size']
        fm_numpix = np.shape(kwargs_data['image_data'])[0]
        adjust_pix = (fm_numpix - numpix)/2
        T = kwargs_data['transform_pix2angle']
        out_file.write('\t\t\t\u0027ra_at_xy_0\u0027: %.10f,\n'%(1.6))
        out_file.write('\t\t\t\u0027dec_at_xy_0\u0027: %.10f,\n'%(1.56))
        out_file.write('\t\t\t\u0027transform_pix2angle\u0027:np.array([[%.8f,%.8f],\n\t\t\t\t[%.8f,%.8f]])\n'%(
            -0.04,0,0,-0.04))
        #out_file.write('\t\t\t\u0027ra_at_xy_0\u0027: %.10f,\n'%((kwargs_data['ra_at_xy_0']+adjust_pix*T[0,0])))
        #out_file.write('\t\t\t\u0027dec_at_xy_0\u0027: %.10f,\n'%(-(kwargs_data['dec_at_xy_0']+adjust_pix*T[1,1])))
        #out_file.write('\t\t\t\u0027transform_pix2angle\u0027:np.array([[%.8f,%.8f],\n\t\t\t\t[%.8f,%.8f]])\n'%(T[0,0],T[0,1],T[1,0],-T[1,1]))
        # close brackets
        out_file.write('\t\t}\n')
        out_file.write('\t},')

    def sky_brightness_helper(self):
        
        background_counts = 19.120 # from here: https://etc.stsci.edu/etc/input/wfc3uvis/imaging/
        ab_zeropoint = self.hst_image.ab_zeropoint()
        return cps2magnitude(background_counts,ab_zeropoint)


    def _write_detector_config(self,out_file):
        """
        Writes block of text with pixel grid dict for paltas config

        Args:
            out_file: opened file to write to
        """
        out_file.write('\n')
        out_file.write('\t\u0027detector\u0027:{\n')
        out_file.write('\t\t\u0027parameters\u0027:{\n')
        out_file.write('\t\t\t\u0027pixel_scale\u0027:0.04,\n')
        out_file.write('\t\t\t\u0027ccd_gain\u0027:1.5,\n')
        out_file.write('\t\t\t\u0027read_noise\u0027:3.0,\n')
        out_file.write('\t\t\t\u0027magnitude_zero_point\u0027:output_ab_zeropoint,\n')
        et = np.average(self.forward_model.multi_band_list[0][0]['exposure_time'])
        out_file.write('\t\t\t\u0027exposure_time\u0027:%.3f,\n'%(et))
        out_file.write('\t\t\t\u0027sky_brightness\u0027:%.3f,\n'%(self.sky_brightness_helper()))
        out_file.write('\t\t\t\u0027num_exposures\u0027:1,\n')
        out_file.write('\t\t\t\u0027background_noise\u0027:None\n')
        # close brackets
        out_file.write('\t\t}\n')
        out_file.write('\t}')

    def _write_drizzle_config(self,out_file):
        """
        Writes block of text with drizzle dict for paltas config

        Args:
            out_file: opened file to write to
        """
        out_file.write('\n')
        out_file.write('\t\u0027drizzle\u0027:{\n')
        out_file.write('\t\t\u0027parameters\u0027:{\n')
        out_file.write('\t\t\t\u0027supersample_pixel_scale\u0027:0.04,\n')
        out_file.write('\t\t\t\u0027output_pixel_scale\u0027:0.04,\n')
        out_file.write('\t\t\t\u0027wcs_distortion\u0027:None,\n')
        out_file.write('\t\t\t\u0027offset_pattern\u0027:[(0.0,0.0),(0.5,0.5)],\n')
        out_file.write('\t\t\t\u0027psf_supersample_factor\u0027:1\n')
        # close brackets
        out_file.write('\t\t}\n')
        out_file.write('\t},')

    def doppelganger_config(self,output_path,no_noise=False,pixel_grid=True,
        emp_PSF=False,drizzle=False):
        """
        Args:
            output_path (string): Path to write .py config file
        """
        
        if self.forward_model is None:
            print("Cannot create config without forward model initialized")
            return
        
        kwargs_model = self.forward_model.kwargs_model
        kwargs_result = self.forward_model.kwargs_result

        config_file = open(output_path,"w")
        # import packages
        self._write_config_imports(config_file)
        # initial calculations
        config_file.write('\n\n')
        self._write_config_setup(config_file,no_noise)
        if emp_PSF:
            self._write_emp_PSF_setup(config_file)
        # Main Deflector
        config_file.write('\n\n')
        config_file.write('config_dict = {')
        self._write_main_deflector(config_file)
        # Source
        self._write_source(config_file)
        # Lens Light
        self._write_lens_light(config_file)
        # Point Source
        self._write_point_source(config_file)
        # Cosmology
        config_file.write('\n')
        config_file.write('\t\u0027cosmology\u0027:{\n')
        config_file.write('\t\t\u0027parameters\u0027:cosmology_params\n')
        config_file.write('\t},')
        # PSF
        if emp_PSF:
            self._write_emp_PSF_config(config_file)
        else:
            self._write_PSF_config(config_file)
        # Pixel Grid
        if pixel_grid:
            self._write_pixel_grid(config_file)
        # Drizzle
        if drizzle:
            self._write_drizzle_config(config_file)
        # Detector
        self._write_detector_config(config_file)

        config_file.write('\n}')
        config_file.close()
        
        
