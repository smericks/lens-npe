"""
Class to store image from paltas simulation, initalized with a paltas config .py
file
"""
from LensSystem.image_base import ImageBase
from paltas.Configs.config_handler import ConfigHandler
import paltas.Sampling.distributions as dist
import numpy as np
from paltas.Utils.lenstronomy_utils import PSFHelper
from lenstronomy.Data.psf import PSF
from lenstronomy.SimulationAPI.data_api import DataAPI
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
import copy
from lenstronomy.Sampling.parameters import Param
from lenstronomy.Sampling.likelihood import LikelihoodModule
from lenstronomy.Sampling.sampler import Sampler
from scipy.stats import norm as norm_dist
import h5py

class PaltasModelImage(ImageBase):
    
    def __init__(self,paltas_config):
        # save some info
        self.config_path = paltas_config
        self.config_handler = ConfigHandler(paltas_config)
        # simulate image based on config file
        self.image_data, self.metadata = self.create_image()
    
    def create_image(self,new_sample=True):
        # load config file and return image that is created
        image,metadata = self.config_handler.draw_image(new_sample=new_sample)
        return image, metadata

    def apply_psf(self,psf_map,flip=False):
        """
        Args:
            psf_map (array): Matrix representing point spread function
            flip (boolean): default=False, if True, mirrors psf map left/right
        """

        image,_ = self.config_handler.draw_image()

        # create PSF Model
        if flip:
            print('flipping PSF map')
            psf_map = np.fliplr(psf_map)
        psf_model_lenstronomy = PSF(psf_type='PIXEL',
            kernel_point_source=psf_map,point_source_supersampling_factor=1)

        # use config handler to create data API
        sample = self.config_handler.get_current_sample()
        kwargs_detector = sample['detector_parameters']
        data_api = DataAPI(numpix=self.config_handler.numpix,**kwargs_detector)

        # build PSF function
        psf_model = PSFHelper(data_api.data_class,psf_model_lenstronomy,
            self.config_handler.kwargs_numerics).psf_model

        # call PSF function on image
        self.image_data = psf_model(image)

    def apply_paltas_psf(self,image,psf_map):

        psf_model_lenstronomy = PSF(psf_type='PIXEL',
            kernel_point_source=psf_map,point_source_supersampling_factor=1)

        # use config handler to create data API
        sample = self.config_handler.get_current_sample()
        kwargs_detector = sample['detector_parameters']
        data_api = DataAPI(numpix=self.config_handler.numpix,**kwargs_detector)

        # build PSF function
        psf_model = PSFHelper(data_api.data_class,psf_model_lenstronomy,
            self.config_handler.kwargs_numerics).psf_model

        # call PSF function on image
        final_image = psf_model(image)

        return final_image

    def image_positions(self):
        """
        Returns:
            ([x_coords],[y_coords]): Two arrays containing ra/dec positions of
                point source images
        """
        kwargs_model, kwargs_params = self.config_handler.get_lenstronomy_models_kwargs(new_sample=False)

        lens_model = LensModel(kwargs_model['lens_model_list'])#,
		#	z_source=kwargs_model['z_source'],
		#	z_source_convention=kwargs_model['z_source_convention'],
		#	lens_redshift_list=kwargs_model['lens_redshift_list'],multi_plane=kwargs_model['multi_plane'])

        point_source_model = PointSource(
			kwargs_model['point_source_model_list'],lens_model=lens_model,
			save_cache=True,fixed_magnification_list=[True])

        image_positions_ps = point_source_model.image_position(kwargs_ps=kwargs_params['kwargs_ps'], kwargs_lens=kwargs_params['kwargs_lens'])

        return [image_positions_ps[0][0],image_positions_ps[1][0]]

    def image_positions_grid(self,data_correction=False):
        """
        Returns:
            ([x_pos],[y_pos]): Two arrays containing grid positions of point 
                source images
        """

        [x_pos,y_pos] = self.image_positions()
        if data_correction:
            x_pos = -(x_pos-0.02)
            y_pos = -(y_pos+0.02)
        sample = self.config_handler.get_current_sample()
        kwargs_detector = sample['detector_parameters']
        data_api = DataAPI(numpix=self.config_handler.numpix,**kwargs_detector)
        grid_x, grid_y = data_api.data_class.map_coord2pix(x_pos,y_pos)
        return grid_x,grid_y


    def im_npe_pred(self,y_pred):
        """
        Given a predicted lens model, return image positions & an image drawn
            with nuisance parameters from existing config + predicted lens model 
            params

        Arguments: 
            y_pred ([float]): [theta_E, gamma1, gamma2, gamma, e1, e2, 
                x_lens,y_lens,x_src,y_src]
        """

        config_dict_copy = copy.deepcopy(self.config_handler.config_dict)
        # overwrite learned params with the learned posteriors
        config_dict_copy['main_deflector']['parameters']['theta_E'] = y_pred[0]
        config_dict_copy['main_deflector']['parameters']['gamma1'] = y_pred[1]
        config_dict_copy['main_deflector']['parameters']['gamma2'] = y_pred[2]
        config_dict_copy['main_deflector']['parameters']['gamma'] = y_pred[3]
        config_dict_copy['main_deflector']['parameters']['e1'] = y_pred[4]
        config_dict_copy['main_deflector']['parameters']['e2'] = y_pred[5]
        config_dict_copy['cross_object']['parameters']['main_deflector:center_x,lens_light:center_x'] = dist.DuplicateScatter(
                    dist=norm_dist(loc=y_pred[6],scale=0).rvs,scatter=0.005)
        config_dict_copy['cross_object']['parameters']['main_deflector:center_y,lens_light:center_y'] = dist.DuplicateScatter(
                    dist=norm_dist(loc=y_pred[7],scale=0).rvs,scatter=0.005)
        config_dict_copy['cross_object']['parameters']['source:center_x,source:center_y,point_source:x_point_source,'+
                    'point_source:y_point_source']=dist.DuplicateXY(x_dist=norm_dist(loc=y_pred[8],scale=0).rvs,
                    y_dist=norm_dist(loc=y_pred[9],scale=0).rvs)
        
        # save and revert at the end
        config_dict_SAVE = copy.deepcopy(self.config_handler.config_dict)
        self.config_handler.config_dict = config_dict_copy
        self.config_handler.sampler = Sampler(config_dict_copy)

        im_npe, _ = self.config_handler.draw_image()
        x_im,y_im = self.image_positions_grid()

        # ok now revert back
        self.config_handler.config_dict = config_dict_SAVE
        self.config_handler.sampler = Sampler(self.config_handler.config_dict)

        return im_npe, x_im, y_im

    def image_amps(self):
        """
        Returns:
            [float]: Array of image amplitudes
        """
        kwargs_model, kwargs_params = self.config_handler.get_lenstronomy_models_kwargs(new_sample=False)

        lens_model = LensModel(kwargs_model['lens_model_list'])
        point_source_model = PointSource(
			kwargs_model['point_source_model_list'],lens_model=lens_model,
			save_cache=True,fixed_magnification_list=[True])

        image_amps = point_source_model.image_amplitude(kwargs_ps=kwargs_params['kwargs_ps'], kwargs_lens=kwargs_params['kwargs_lens'])

        return image_amps

    def alter_sample_psf(self,psf_kernel):
        """
        Changes values stored in config handler's current sample
        """
        self.config_handler.sample['psf_parameters']['kernel_point_source'] = psf_kernel

    def run_lenstronomy_MCMC(self,chains_path,include_astrometry=False,
                             num_samples=10000,burn_in=1000):
        """
        Uses paltas config to simulate an image,
        and then run a lenstronomy MCMC to explore shape of forward model
        posterior given the image

        Args:
            chains_path (string): path to save chains in a .h5 file
            include_astrometry (bool): If true, include an extra term in the 
                likelihood to penalize point source positions in the wrong place
            num_samples (int): number samples after burn-in removed
            burn_in (int): number of samples thrown away at beginning of chain
        """

        # simulate the image
        image,_ = self.config_handler.draw_image()
        # retrieve ground truth & PSF info
        sample = self.config_handler.get_current_sample()
        kwargs_model_paltas, kwargs_params_paltas = (
            self.config_handler.get_lenstronomy_models_kwargs(new_sample=False))
        

        kwargs_model = {'lens_model_list': kwargs_model_paltas['lens_model_list'], 
            'source_light_model_list': kwargs_model_paltas['source_light_model_list'],
            'lens_light_model_list': kwargs_model_paltas['lens_light_model_list'],
            'point_source_model_list': kwargs_model_paltas['point_source_model_list']} 

        # Define the prior (follow bounds of interim training prior)
        # DEFLECTOR (assumes EPL + SHEAR)
        kwargs_fixed_lens = kwargs_fixed_lens = [{},{'ra_0': 0,'dec_0': 0}]
        kwargs_lower_lens = [{'theta_E':0.5,'e1':-0.75,'e2':-0.75,'gamma':1.5,
                      'center_x':-0.2,'center_y':-0.2},
                     {'gamma1':-0.6,'gamma2':-0.6}]
        kwargs_upper_lens = [{'theta_E':1.5,'e1':0.75,'e2':0.75,'gamma':2.5,
                      'center_x':0.2,'center_y':0.2},
                     {'gamma1':0.6,'gamma2':0.6}]

        # SOURCE (assumes SERSIC_ELLIPSE)
        kwargs_fixed_source = [{}]
        kwargs_lower_source = [{'R_sersic':0.001,'n_sersic':0.5,
                                'e1':-0.75,'e2':-0.75,
                                'center_x':-0.2,'center_y':-0.2}]
        kwargs_upper_source = [{'R_sersic':2.5,'n_sersic':8.,
                                'e1':0.75,'e2':0.75,
                                'center_x':0.2,'center_y':0.2}]
        
        # LENS LIGHT (assumes SERSIC_ELLIPSE)
        kwargs_fixed_lens_light = [{}]
        kwargs_lower_lens_light = [{'R_sersic':0.001,'n_sersic':0.5,
                                    'e1':-0.75,'e2':-0.75,
                                    'center_x':-0.2,'center_y':-0.2}]
        kwargs_upper_lens_light = [{'R_sersic':3.,'n_sersic':7.,
                                    'e1':0.75,'e2':0.75,
                                    'center_x':0.2,'center_y':0.2}]
        
        # POINT SOURCE (assumes SOURCE_POSITION)
        kwargs_fixed_ps = [{}]
        kwargs_lower_ps = [{'ra_source':-0.2,'dec_source':-0.2}]
        kwargs_upper_ps = [{'ra_source':0.2,'dec_source':0.2}]

        # Join source & point source center
        kwargs_constraints = {'joint_source_with_point_source': 
            [[0,0,{'center_x':'ra_source','center_y':'dec_source'}]]}


        param = Param(kwargs_model=kwargs_model,kwargs_fixed_lens=kwargs_fixed_lens,
              kwargs_fixed_source=kwargs_fixed_source,
              kwargs_fixed_lens_light=kwargs_fixed_lens_light,
              kwargs_fixed_ps=kwargs_fixed_ps,
              kwargs_lower_lens=kwargs_lower_lens,
              kwargs_lower_source=kwargs_lower_source,
              kwargs_lower_lens_light=kwargs_lower_lens_light,
              kwargs_lower_ps=kwargs_lower_ps,
              kwargs_upper_lens=kwargs_upper_lens,
              kwargs_upper_source=kwargs_upper_source,
              kwargs_upper_lens_light=kwargs_upper_lens_light,
              kwargs_upper_ps=kwargs_upper_ps,
              **kwargs_constraints)
        
        # Define image data information for likelihood calculation
        # consider not hard-coding exposure time, background_rms...
        kwargs_data = {'image_data': image,
               'background_rms': 0.006,
               'exposure_time': 1360.522,
               'ra_at_xy_0': -1.58,  
               'dec_at_xy_0': -1.58,
               'transform_pix2angle': np.array([[0.04, 0], [0, 0.04]])}

        # assume we know PSF exactly
        kwargs_psf = sample['psf_parameters']
        # no supersampling
        kwargs_numerics = {'supersampling_factor': 1,
                        'supersampling_convolution': False}
        
        single_band = [[kwargs_data,kwargs_psf,kwargs_numerics]]
        kwargs_data_joint = {'multi_band_list': single_band, 
                             'multi_band_type': 'multi-linear'}

        kwargs_likelihood = {}
        if include_astrometry:
            kwargs_likelihood = {'astrometric_likelihood':True}

        likelihoodModule = LikelihoodModule(kwargs_data_joint, kwargs_model, param_class=param,
                                            **kwargs_likelihood)
        
        # where to initialize sampling
        kwargs_lens_init = copy.deepcopy(kwargs_params_paltas['kwargs_lens'])
        kwargs_lens_init[1].pop('ra_0')
        kwargs_lens_init[1].pop('dec_0')
 
        kwargs_source_init = copy.deepcopy(kwargs_params_paltas['kwargs_source'])
        kwargs_source_init[0].pop('amp')

        kwargs_lens_light_init = copy.deepcopy(
            kwargs_params_paltas['kwargs_lens_light'])
        kwargs_lens_light_init[0].pop('amp')

        # disregard any other PS params (i.e. mag_pert)
        kwargs_ps_init = [{'ra_source':kwargs_params_paltas['kwargs_ps'][0]['ra_source'],
                            'dec_source':kwargs_params_paltas['kwargs_ps'][0]['dec_source']}]

        # initial spread in parameter estimation #
        kwargs_lens_sigma = [{'theta_E': 0.1, 'e1': 0.01, 'e2': 0.01, 
                            'gamma': .05, 'center_x': 0.005, 'center_y': 0.005},
                            {'gamma1': 0.01, 'gamma2': 0.01}]
        kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': .05, 
                                'e1':0.01, 'e2':0.01,
                                'center_x': 0.005, 'center_y': 0.005}]
        kwargs_lens_light_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.05, 
                                    'e1': 0.01, 'e2': 0.01, 
                                    'center_x': 0.005, 'center_y': 0.005}]
        kwargs_ps_sigma = [{'ra_source':0.005,'dec_source':0.005}]

        param_init = param.kwargs2args(kwargs_lens_init, kwargs_source_init, 
                                    kwargs_lens_light_init, kwargs_ps=kwargs_ps_init)
        param_sigma = param.kwargs2args(kwargs_lens_sigma, kwargs_source_sigma, 
                                        kwargs_lens_light_sigma, kwargs_ps=kwargs_ps_sigma)
        
        sampler = Sampler(likelihoodModule=likelihoodModule)
        samples, ln_likelihood = sampler.mcmc_emcee(
            n_walkers=100, n_run=num_samples, n_burn=burn_in, 
            mean_start=param_init, sigma_start=param_sigma)
        
        if chains_path[-3:] != '.h5':
            raise ValueError('chains_path must end with .h5')
        h5f = h5py.File(chains_path, 'w')
        h5f.create_dataset('samples', data=samples)
        h5f.close()

        return



