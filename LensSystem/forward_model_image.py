"""
Class to store forward model image, initalized with a results.txt file
"""
from LensSystem.image_base import ImageBase
import pickle
from lenstronomy.ImSim.MultiBand.multi_linear import MultiLinear
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Plots.model_plot import ModelPlot
import numpy as np
import copy

class ForwardModelImage(ImageBase):
    """
    Args:
        results_file (string): results.txt file from forward modeling with 
            lenstronomy (compressed with pickle)
    """


    def __init__(self,results_file,make_image=True):

        super().__init__()

        f = open(results_file,'rb')
        (self.multi_band_list, self.kwargs_model, self.kwargs_result, 
            self.image_likelihood_mask_list) = pickle.load(f)
        f.close()

        # psf_error_map is not backwards compatible!!
        """Any psf_error_map previously being used needs to be multiplied by the 
            kernel_point_source^2.
            https://github.com/lenstronomy/lenstronomy/pull/424
        """
        self.multi_band_list[0][1]['psf_error_map'] = (
            self.multi_band_list[0][1]['psf_error_map']*(
                self.multi_band_list[0][1]['kernel_point_source']**2))

        # save some information
        self.psf_map = self.multi_band_list[0][1]['kernel_point_source']
        self.results_file = results_file

        if make_image:

            # we must use MultiLinear method upon initialization so that linear
            # solving for amplitudes is done
            self.image_data = self.image_MultiLinear()

            # save lens_model and point_source_model
            self.lens_model = LensModel(self.kwargs_model['lens_model_list'])
            if 'fixed_magnification_list' in self.kwargs_model:
                self.point_source_model = PointSource(
                    [self.kwargs_model['point_source_model_list'][0]],
                    lens_model=self.lens_model,
                    fixed_magnification_list=
                    self.kwargs_model['fixed_magnification_list'])
            else:
                self.point_source_model = PointSource(
                    [self.kwargs_model['point_source_model_list'][0]],
                    lens_model=self.lens_model)

    def image_MultiLinear(self):

        im_sim = MultiLinear([self.multi_band_list[0]],
            self.kwargs_model,likelihood_mask_list=
            [self.image_likelihood_mask_list[0]])
        model, _, _, params = im_sim.image_linear_solve(
            kwargs_lens=self.kwargs_result['kwargs_lens'],kwargs_source=
                self.kwargs_result['kwargs_source'],
            kwargs_lens_light=self.kwargs_result['kwargs_lens_light'],
                kwargs_ps=self.kwargs_result['kwargs_ps'])
        
        return model[0]

    def image_no_PSF(self):
        kwargs_psf_copy = self.multi_band_list[0][1]
        self.multi_band_list[0][1] = {'psf_type':'NONE'}
        image = self.image_MultiLinear()
        self.multi_band_list[0][1] = kwargs_psf_copy
        return image

    def image_ImageModel(self,no_PSF=False,custom_data_model=None):

        if custom_data_model is not None:
            data_model = custom_data_model
        else:
            # Pixel Grid / Detector
            data_model = ImageData(**self.multi_band_list[0][0])

        # Point Spread Function (PSF)
        if no_PSF:
            psf_model = PSF(psf_type='NONE')
        else:
            psf_model = PSF(**self.multi_band_list[0][1])

        # Lens Mass
        lens_model = self.lens_model
        kwargs_lens_list = self.kwargs_result['kwargs_lens']

        # Source Light
        source_light_model = LightModel(
            [self.kwargs_model['source_light_model_list'][0]])
        kwargs_source_list = [self.kwargs_result['kwargs_source'][0]]

        # Lens Light
        lens_light_model = LightModel(
            [self.kwargs_model['lens_light_model_list'][0]])
        kwargs_lens_light_list = [self.kwargs_result['kwargs_lens_light'][0]]

        # Point Source Light
        point_source_model = self.point_source_model
        kwargs_ps_list = self.kwargs_result['kwargs_ps']

        # Numerics
        kwargs_numerics = self.multi_band_list[0][2]

        # Create ImageModel
        image_model = ImageModel(data_model,psf_model,
            lens_model,source_light_model,lens_light_model,
            point_source_model,kwargs_numerics=kwargs_numerics)

        # Draw image
        image = image_model.image(kwargs_lens_list,kwargs_source_list,
            kwargs_lens_light_list,kwargs_ps_list)

        return image


    def image_positions(self):
        """
        Returns:
            ([x_coords],[y_coords]): Two arrays containing ra/dec positions of
                point source images
        """

        image_positions_ps = self.point_source_model.image_position(
            kwargs_ps=self.kwargs_result['kwargs_ps'], 
            kwargs_lens=self.kwargs_result['kwargs_lens'])

        return [image_positions_ps[0][0],image_positions_ps[1][0]]

    # modify point source model
    def modify_ps(self,ps_model,kwargs_ps):
        """
        Args:
            ps_model (string): Type of lenstronomy PS model
            kwargs_ps (dict): Dict of params to pass to lenstronomy PS model
        """

        # if changing from LENSED_POSIITON to SOURCE_POSITION, we must add
        # fixed_magnification_list to model kwargs
        if (self.kwargs_model['point_source_model_list'][0] == 'LENSED_POSITION'
            and ps_model == 'SOURCE_POSITION'):

            self.kwargs_model['fixed_magnification_list'] = [True]

        # if switching the opposite way, remove fixed_magnification_list
        if (self.kwargs_model['point_source_model_list'][0] == 'SOURCE_POSITION'
            and ps_model == 'LENSED_POSITION'):

            self.kwargs_model.pop('fixed_magnification_list',None)

        self.kwargs_model['point_source_model_list'][0] = ps_model
        self.kwargs_result['kwargs_ps'] = [kwargs_ps]

        # re-save self.point_source_model
        if 'fixed_magnification_list' in self.kwargs_model:
            self.point_source_model = PointSource(
                [self.kwargs_model['point_source_model_list'][0]],
                lens_model=self.lens_model, 
                fixed_magnification_list=
                self.kwargs_model['fixed_magnification_list'])
        else:
            self.point_source_model = PointSource(
                [self.kwargs_model['point_source_model_list'][0]],
                lens_model=self.lens_model)

        # re-draw image with modified PS params
        self.image_data = self.image_ImageModel()

    def plot_caustics_source(self,ax):
        bandplot = ModelPlot(self.multi_band_list,self.kwargs_model,self.kwargs_result)
        bandplot.source_plot(band_index=0,ax=ax,numPix=15,deltaPix_source=0.04,with_caustics=True)

    def plot_caustics_images(self,ax):
        bandplot = ModelPlot(self.multi_band_list,self.kwargs_model,self.kwargs_result)
        bandplot.deflection_plot(band_index=0,ax=ax,with_caustics=True)

    # compute source amplitude from 4 image amplitudes
    def image_amp_to_source_amp(self,debug=False):
        """
        Returns:
            point_amp (float): Amplitude of source in source plane
            mag_corrections (list):  Magnification corrections that must be 
                applied to each image when switching from the 4 image amplitudes
                to the single source amplitude. This ensures final image 
                amplitudes are exactly the same.
        """
        kwargs_model = self.kwargs_model

        # check that image information exists
        if kwargs_model['point_source_model_list'][0] != 'LENSED_POSITION':
            print("Must use LENSED_POSITION model to use this function")
            return

        # retrieve image amps & image positions
        image_amps = self.kwargs_result['kwargs_ps'][0]['point_amp']
        x_img = np.array(self.kwargs_result['kwargs_ps'][0]['ra_image'])
        y_img = np.array(self.kwargs_result['kwargs_ps'][0]['dec_image'])
        
        # compute magnifications at image positions
        kwargs_lens=self.kwargs_result['kwargs_lens']
        image_magnifications = self.lens_model.magnification(x=x_img, y=y_img, 
            kwargs=kwargs_lens)

        # demagnify & avg. values
        point_amp = np.average(image_amps/np.abs(image_magnifications))

        # compute magnification corrections
        new_image_amps = np.abs(image_magnifications)*point_amp

        # need to re-order corrections based on new order of images
        # TODO: revert lens eqn solver kwargs
        data_model = ImageData(**self.multi_band_list[0][0])
        kwargs_lens_eqn_solver = {
            'search_window':data_model.width[0],
            'min_distance':data_model.pixel_width
        }
        point_source = PointSource(
                ['SOURCE_POSITION'],
                lens_model=self.lens_model, 
                fixed_magnification_list=[True],
                kwargs_lens_eqn_solver=kwargs_lens_eqn_solver)
        if debug:
            ra_ps,dec_ps = self.image_pos_to_source_pos(verbose=True)
        else:
            ra_ps,dec_ps = self.image_pos_to_source_pos()
        
        kwargs_ps = {
            'source_amp':point_amp,
            'ra_source':ra_ps,
            'dec_source':dec_ps
        }

        if debug:
            return point_source,kwargs_ps

        x_img_new, y_img_new = point_source.image_position(kwargs_ps=[kwargs_ps], 
            kwargs_lens=kwargs_lens)

        mag_corrections = []
        for i in range(0,len(new_image_amps)):
            idx = np.where(np.abs(x_img - x_img_new[0][i] ) < 0.01)[0][0]
            old_amp = image_amps[idx]
            new_amp = new_image_amps[idx]
            mag_corrections.append(old_amp / new_amp)

        return point_amp, mag_corrections
    
    # compute source position from 4 image positions of LENSED_POSITION model
    def image_pos_to_source_pos(self,verbose=False):
        """
        Returns:
            ra_ps,dec_ps (float,float): ra/dec position of point source in the 
                source plane w.r.t the center of the image. 
            verbose (boolean): If True, prints each image's solution & averaged
                solution.
        """
        kwargs_params = self.kwargs_result

        image_positions_ps = self.point_source_model.image_position(
                kwargs_ps=kwargs_params['kwargs_ps'],
                kwargs_lens=kwargs_params['kwargs_lens'])
        x_img = image_positions_ps[0][0]
        y_img = image_positions_ps[1][0]

        # computing source position from image position
        x_beta, y_beta = self.lens_model.ray_shooting(x=x_img, y=y_img,
            kwargs=kwargs_params['kwargs_lens'])

        if verbose: 
            print("Individual Solutions:")
            for i in range(0,len(x_beta)):
                print("(",x_beta[i],",",y_beta[i],")")

        ra_ps = np.average(x_beta)
        dec_ps = np.average(y_beta)

        if verbose: 
            print("Averaged Solution:")
            print("(",ra_ps,",",dec_ps,")")

        return ra_ps,dec_ps

    # retrieve coordinates for center of host/source galaxy
    def host_center_pos(self):
        """
        Returns:
            ra_ps,dec_ps (float,float): ra/dec position of host galaxy in the 
                source plane w.r.t the center of the image. 
        """
        ra_host = self.kwargs_result['kwargs_source'][0]['center_x']
        dec_host = self.kwargs_result['kwargs_source'][0]['center_y']

        return ra_host,dec_host

    # calculate noise map
    def error_map_divisor(self,numpix=None):
        """
        Returns: 
            2d array of float: sigma of error due to poisson noise, background noise, 
            and psf error
        """
        from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel
        multiband = SingleBandMultiModel(multi_band_list=[self.multi_band_list[0]], kwargs_model=self.kwargs_model, 
            likelihood_mask_list=[self.image_likelihood_mask_list[0]],band_index=0,linear_solver=False)

        model, error_map, _, _ = multiband.image_linear_solve(inv_bool=True,**self.kwargs_result)
        # reduced residuals returns: (model - Data)/sigma_error
        inverse_error = multiband.reduced_residuals(model, error_map=error_map)/(model - multiband.Data.data)
        
        if numpix is not None:
            inverse_error = self.center_crop(inverse_error,numpix)

        return inverse_error
    

    # adjust all x/y coordinates
    def adjust_xy_coords(x_adjust=0.,y_adjust=0.):
        """ adjust all (x,y) coordinates by additive factor x(y)_adjust
        
        Args:
            x_adjust (float)
            y_adjust (float)

        Returns:
            nothing, adjusts objects in place
        """


        # go through self.kwargs_results and change everything

        # main deflector mass
        # lens light
        # source light
        # point source light
        
