# began with: https://hst-docs.stsci.edu/hstdhb/files/60242993/67928432/1/1594401324445/HSTDHB.pdf

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from astropy.visualization import simple_norm
import pandas as pd
from PIL import Image
import warnings
import matplotlib as mpl
import os
import shutil
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util.data_util import cps2magnitude, magnitude2cps
import pickle
from scipy.ndimage import rotate
from scipy.stats import multivariate_normal
from paltas.Configs.config_handler import ConfigHandler
from paltas.Utils.cosmology_utils import get_cosmology, absolute_to_apparent
from paltas.Analysis import posterior_functions
import corner
import seaborn as sns
import sys
from matplotlib.lines import Line2D
import h5py

# define some global colors to standardize plotting
palette = sns.color_palette('muted').as_hex()
COLORS = {
    'prior':palette[7],
    'hyperparam':palette[3],
    'hyperparam_narrow':palette[1],
    'unweighted_NPE':palette[0],
    'reweighted_NPE':palette[4]
}



"""
Functions:
    print_header(file_name)
    header_ab_zeropoint(hdu)
    amp_2_apparent_mag(model_name,model_kwargs,ab_zeropoint)
    ps_amp_2_apparent_mag(amp,ab_zeropoint)
    ps_image2source_magnitude(magnitudes,magnifications,ab_zeropoint)
    log_norm(data,min_cutoff=1e-6)
    residuals_norm(im_diff)
    load_psf_from_pickle(results_file)
    matrix_plot(image_folder,df,indices,dim,save_name,PlotCenter=True)
    matrix_plot_from_npy(file_list,names,dim,save_name)
    plot_compare_prior_datset(params_list,df,priors_dict,title,
                        df_suffix='_STRIDES_median',special_set=None)
    prior_corner_plot(prior_dict,N)
    prior_corner_plot_slow(paltas_config_file,N)
    plot_coverage(y_pred,y_test,std_pred,parameter_names,
	    fontsize=20,show_error_bars=True,n_rows=4,bin_min=0.9,bin_max=250)
    table_metrics(y_pred,y_test,std_pred,outfile)
"""

#######################
# Ignore wcs warnings
#######################
warnings.filterwarnings("ignore")

#######################
# Print entire header
#######################
def print_header(file_name):
    """
    Args:
        file_name: path to .fits file
    """
    with fits.open(file_name) as hdu:
        print(hdu[0].header)

#########################################
# Get AB zeropoint magnitude from header
#########################################
# sources: 
# https://hst-docs.stsci.edu/acsdhb/chapter-5-acs-data-analysis/5-1-photometry
# https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
def header_ab_zeropoint(hdu):
    """
    Args: 
        hdu: opened .fits file object
    """
    
    # mean flux density (erg*cm^-2*s^-1*angstrom)
    photflam = hdu[0].header['PHOTFLAM']
    # pivot wavelength (angstrom)
    photplam = hdu[0].header['PHOTPLAM']

    return -2.5*np.log10(photflam) -5*np.log10(photplam) - 2.408

##################################
# Amplitude to Apparent Magnitude
##################################
# note amplitude is surface brightness at R_sersic
def amp_2_apparent_mag(model_name,model_kwargs,ab_zeropoint):
    """
    Args: 
        model_name (string): lenstronomy SOURCE model name
        model_kwargs (dict): lenstronomy source model parameters
        ab_zeropoint (float): AB magnitude zeropoint
    """
    # get cps from amplitude
    sersic_model = LightModel([model_name])
    # returns list
    integrated_flux = sersic_model.total_flux([model_kwargs])[0]
    # get magnitude from cps
    magnitude = cps2magnitude(integrated_flux,ab_zeropoint)
    return magnitude

##############################################
# Point Source Amplitude to Apparent Magnitude
##############################################
def ps_amp_2_apparent_mag(amp,ab_zeropoint):
    """
    Args:
        amp (float): Amplitude in cps of point source
        ab_zeropoint (float): AB magnitude zeropoint
    Returns:
        mag (float): Apparent magnitude of point source
    """
    mag = cps2magnitude(amp,ab_zeropoint)
    return mag

#####################################################################
# Point source image magnitudes + magnifications to source magnitude
#####################################################################
def ps_image2source_magnitude(magnitudes,magnifications,ab_zeropoint):
    """
    Args: 
        magnitudes ([float]): AB magnitudes of point source images
        magnifications ([float]): Magnification of each point source image
        ab_zeropoint (float): AB magnitude zeropoint

    Returns: 
        magnitude_source (float): apparent magnitude of unlensed point source
    """
    # compute counts per second from image magnitudes
    cps_magnified = []
    for m in magnitudes:
        cps = magnitude2cps(m,ab_zeropoint)
        cps_magnified.append(cps)
    cps_magnified = np.asarray(cps_magnified)

    # de-magnify counts per second
    magnifications = np.asarray(magnifications)
    cps_demagnified = cps_magnified / np.abs(magnifications)

    # convert de-magnified cps back to magnitudes
    avg_cps = np.average(cps_demagnified)
    magnitude_source = cps2magnitude(avg_cps,ab_zeropoint)

    return magnitude_source

#############
# log norm
#############
def log_norm(data,min_cutoff=1e-6):

    norm = simple_norm(data,stretch='log',min_cut=min_cutoff)
    return norm

########################################
# Create blue to red norm for residuals
########################################
def residuals_norm(im_diff):

    if np.amin(im_diff) > 0:
        norm = colors.TwoSlopeNorm(vmin=-0.001,vcenter=0,vmax=np.amax(im_diff))
    elif np.amax(im_diff) < 0:
        norm = colors.TwoSlopeNorm(vmin=np.amin(im_diff), vcenter=0, vmax=0.001)
    else:
        norm = colors.TwoSlopeNorm(vmin=np.amin(im_diff), vcenter=0, vmax=np.amax(im_diff))

    return norm

#######################################
# Load psf map from pickle results file
#######################################

def load_psf_from_pickle(results_file):
    f = open(results_file,'rb')
    (multi_band_list, kwargs_model, kwargs_result, 
            image_likelihood_mask_list) = pickle.load(f)
    f.close()
    return multi_band_list[0][1]['kernel_point_source']

#################################
# Function to create matrix plot
#################################
def matrix_plot(image_folder,df,indices,dim,save_name,PlotCenter=True,
                show_one_arcsec=True):
    """
    Args: 
        image_folder (string): Name/path of folder storing .fits files
        df (pandas dataframe): Dataframe containing all info from lens catalog 
            .csv file
        indices (numpy array): List of indices of which systems to plot
        dim (int,int): Tuple of (number rows, number columns), defines shape of 
            matrix plotted
        save_name (string): Filename to save final image to
        PlotCenter (boolean): If True, plots small red star in center of image
        show_one_arcsec (boolean): If True, plots 1 arcsec bar in top left corner
    """

    # extract info from dataframe
    filenames = df['file_prefix'].to_numpy()
    system_names = df['name'].to_numpy()
    ra_adjust = df['ra_adjust'].to_numpy()
    dec_adjust = df['dec_adjust'].to_numpy()
    cutout_size = df['cutout_size'].to_numpy()

    # append -1 to indices to fill empty spots manually
    if dim[0]*dim[1] > len(indices):
        # num empty spots to fill
        num_fill = dim[0]*dim[1] - len(indices)
        # append -1 num_fill times
        for m in range(0,num_fill):
            indices = np.append(indices,-1)

    # edge case: not enough spaces given for # of images
    if dim[0]*dim[1] < len(indices):
        print("Matrix not big enough to display all lens images." + 
            "Retry with larger matrix dimensions. Exiting matrix_plot()")
        return

    # initalize loop variables
    file_counter = 0
    completed_rows = []
    # TODO: check to see if this folder already exists
    os.mkdir('intermediate_temp')

    # prevent matplotlib from showing intermediate plots
    backend_ =  mpl.get_backend() 
    mpl.use("Agg")  # Prevent showing stuff

    # iterate through matrix & fill each image
    for i in range(0,dim[0]):
        row = []
        for j in range(0,dim[1]):
            
            file_idx = int(indices[file_counter])

            # fill empty spot (index = -1)
            if file_idx == -1:
                plt.matshow(np.ones((100,100)),cmap='Greys_r',dpi=300)
                plt.axis('off')
                plt.savefig(intm_name,bbox_inches='tight',pad_inches=0)
                img_data = np.asarray(Image.open(intm_name))
                row.append(img_data)
                continue
            
            to_open = image_folder + filenames[file_idx] + '_F814W_drc_sci.fits' 

            with fits.open(to_open) as hdu:
                # keep track of which system currently working with
                #print(system_names[file_idx])
                # read in data 
                data = hdu[0].data
                # reflect dec axis for correct E/W convention
                data = np.flip(data,axis=0)
                wcs = WCS(hdu[0].header)
                ra_targ = hdu[0].header['RA_TARG']
                dec_targ = hdu[0].header['DEC_TARG']
                #print("ra: %.3f, dec: %.3f"%(ra_targ,dec_targ))

                # print zeropoint
                #print("zeropoint ", file_idx, ": ", header_ab_zeropoint(hdu))

                # apply any needed adjustments to center of lens system
                ra_targ = ra_targ + ra_adjust[file_idx]
                dec_targ = dec_targ + dec_adjust[file_idx]

                # retrieve array coordinates of lens center
                x_targ,y_targ = wcs.world_to_array_index(SkyCoord(ra=ra_targ,
                        dec=dec_targ,unit="deg"))
                
                # transform x_targ to reflected dec coordinates
                num_x_pixels = hdu[0].header['NAXIS2']
                x_targ = int(x_targ - 2*(x_targ - (num_x_pixels/2)))

                # find out # arseconds per pixel
                arcseconds_per_pixel = hdu[0].header['D003SCAL']
                #print("%.3f arcseconds per pixel"%(arcseconds_per_pixel))

            # retreive offset and crop data
            offset = int(cutout_size[file_idx]/2)
            cropped_data = data[x_targ-offset:x_targ+offset,
                y_targ-offset:y_targ+offset]

            # normalize data using log and min cutoff 
            norm = simple_norm(cropped_data,stretch='log',min_cut=1e-6)
        
            # create individual image using plt library
            plt.figure(dpi=300)
            plt.matshow(cropped_data,cmap='viridis',norm=norm)
            if PlotCenter:
                plt.scatter(offset,offset,edgecolors='red',marker='*',
                    facecolors='none',s=100)
            # annotate system name
            plt.annotate(system_names[file_idx],(2*offset - offset/8,offset/6),color='white',
                fontsize=17,horizontalalignment='right')
            # show size of 1 arcsec
            if show_one_arcsec:
                if file_counter == 0:
                    plt.plot([offset/6,offset/6],[offset/8,
                        offset/8 + (1/arcseconds_per_pixel)],color='white')
                    plt.annotate('1"',(offset/6 +2,offset/2),color='white',fontsize=20)
            plt.axis('off')

            # save intermediate file, then read back in as array, and save to row
            intm_name = ('intermediate_temp/'+ df['file_prefix'].to_numpy()[file_idx]
                +'.png')
            plt.savefig(intm_name,bbox_inches='tight',pad_inches=0)
            img_data = np.asarray(Image.open(intm_name))
            plt.close()
            row.append(img_data)
            
            # manually iterate file index
            file_counter += 1

        # stack each row horizontally in outer loop
        # edge case: one column
        if dim[1] == 1:
            build_row = row[0]
        else:
            build_row = np.hstack((row[0],row[1]))
        
        if dim[1] > 2:
            for c in range(2,dim[1]):
                build_row = np.hstack((build_row,row[c]))
        completed_rows.append(build_row)

    # reset matplotlib s.t. plots are shown
    mpl.use(backend_) # Reset backend
    
    # clean up intermediate files
    shutil.rmtree('intermediate_temp')

    # stack rows to build final image
    # edge case: one row
    if dim[0] == 1:
        final_image = completed_rows[0]
    else:
        final_image = np.vstack((completed_rows[0],completed_rows[1]))

    if dim[0] > 2:
        for r in range(2,dim[0]):
            final_image = np.vstack((final_image,completed_rows[r]))

    # plot image and save
    plt.figure(figsize=(2*dim[1],2*dim[0]),dpi=300)
    plt.imshow(final_image)
    plt.axis('off')
    plt.savefig(save_name,bbox_inches='tight')
    plt.show()

def matrix_plot_from_folder(folder_path,save_path):
    """Takes in a folder with .npy images and returns a default grid of images
    For more flexibility, see matrix_plot_from_npy
    
    Args:
        folder_path (string): path to folder w/ .npy images created by paltas
        save_path (string): path to save final image to
    Returns:

    """
    file_list = []
    names=[]
    for i in range(0,100):
        file_list.append(folder_path+'image_%07d.npy'%(i))
        names.append(str(i))

    matrix_plot_from_npy(file_list[:40],names,(4,10),save_path,annotate=False)

def matrix_plot_from_h5(file_name,dim,save_name):
    """
    Args: 
        file_name (string): path to .h5 file storing images
        dim (int,int): Tuple of (number rows, number columns), defines shape of 
            matrix plotted
        save_name (string): Filename to save final image to
    """

    # load in .h5 file
    f = h5py.File(file_name, "r")
    f_data = f['data'][()]
    print('size of dataset: ',f_data.shape)

    # TODO: check to see if this folder already exists
    os.mkdir('intermediate_temp')

    # prevent matplotlib from showing intermediate plots
    backend_ =  mpl.get_backend() 
    mpl.use("Agg")  # Prevent showing stuff

    file_counter = 0
    completed_rows = []
    offset = 40
    for i in range(0,dim[0]):
        row = []
        for j in range(0,dim[1]):
            cropped_data = f_data[file_counter]
            # normalize data using log and min cutoff 
            norm = simple_norm(cropped_data,stretch='log',min_cut=1e-6)
        
            # create individual image using plt library
            plt.figure(dpi=300)
            plt.matshow(cropped_data,cmap='viridis',norm=norm)
            plt.axis('off')

            # save intermediate file, then read back in as array, and save to row
            intm_name = ('intermediate_temp/'+ str(file_counter)
                +'.png')
            plt.savefig(intm_name,bbox_inches='tight',pad_inches=0)
            img_data = np.asarray(Image.open(intm_name))
            plt.close()
            row.append(img_data)
            # manually iterate file index
            file_counter += 1

        # stack each row horizontally in outer loop
        # edge case: one column
        if dim[1] == 1:
            build_row = row[0]
        else:
            build_row = np.hstack((row[0],row[1]))

        if dim[1] > 2:
            for c in range(2,dim[1]):
                build_row = np.hstack((build_row,row[c]))
                
        completed_rows.append(build_row)

    # reset matplotlib s.t. plots are shown
    mpl.use(backend_) # Reset backend
    
    # clean up intermediate files
    shutil.rmtree('intermediate_temp')

    # stack rows to build final image
    # edge case: one row
    if dim[0] == 1:
        final_image = completed_rows[0]
    else:
        final_image = np.vstack((completed_rows[0],completed_rows[1]))

    if dim[0] > 2:
        for r in range(2,dim[0]):
            final_image = np.vstack((final_image,completed_rows[r]))

    # plot image and save
    plt.figure(figsize=(2*dim[1],2*dim[0]),dpi=300)
    plt.imshow(final_image)
    plt.axis('off')
    plt.savefig(save_name,bbox_inches='tight')
    plt.show()

###############################
# Matrix plot from numpy files
###############################
def matrix_plot_from_npy(file_list,names,dim,save_name,annotate=False,
        show_one_arcsec=False,rotate_data=False):
    """
    Args: 
        file_list ([string]): paths of .npy files
        names ([string]): labels for images
        dim (int,int): Tuple of (number rows, number columns), defines shape of 
            matrix plotted
        save_name (string): Filename to save final image to
        annotate (bool): If True, plots label at top of image
        show_one_arcsec (bool): If True, plots 1" bar at top left of each image
        rotate_data (bool): If True, mirrors image & rotates 180 deg. (to account
            for a mismatch in ra/dec convention)
    """

    # edge case: not enough spaces given for # of images
    if dim[0]*dim[1] < len(file_list):
        print("Matrix not big enough to display all lens images." + 
            "Retry with larger matrix dimensions. Exiting matrix_plot()")
        return

    # TODO: check to see if this folder already exists
    os.mkdir('intermediate_temp')

    # prevent matplotlib from showing intermediate plots
    backend_ =  mpl.get_backend() 
    mpl.use("Agg")  # Prevent showing stuff

    file_counter = 0
    completed_rows = []
    offset = 40
    for i in range(0,dim[0]):
        row = []
        for j in range(0,dim[1]):
            plt.figure(dpi=300)
            if file_list[file_counter] is None:
                cropped_data = np.zeros((80,80))
            else:
                cropped_data = np.load(file_list[file_counter])
                if rotate_data:
                    cropped_data = np.rot90(cropped_data,2)
                # normalize data using log and min cutoff 
            norm = simple_norm(cropped_data,stretch='log',min_cut=1e-6)
        
            # create individual image using plt library
            if names[file_counter] is not None:
                plt.matshow(cropped_data,cmap='viridis',norm=norm)
            else:
                plt.matshow(cropped_data,cmap='binary')

            if annotate and file_list[file_counter] is not None:
                # annotate system name
                plt.annotate(names[file_counter],(2*offset - offset/8,offset/6),color='white',
                    fontsize=17,horizontalalignment='right')
            if show_one_arcsec and file_list[file_counter] is not None:
                if file_counter == 0:
                    # show size of 1 arcsec
                    plt.plot([offset/6,offset/6],[offset/8,
                        offset/8 + (1/0.04)],color='white')
                    plt.annotate('1"',(offset/6 +2,offset/2),color='white',fontsize=20)
            
            plt.axis('off')

            # save intermediate file, then read back in as array, and save to row
            if names[file_counter] is None:
                intm_name = 'intermediate_temp/none.png'
            else:
                intm_name = ('intermediate_temp/'+ names[file_counter]
                    +'.png')
            plt.savefig(intm_name,bbox_inches='tight',pad_inches=0)
            img_data = np.asarray(Image.open(intm_name))
            plt.close()
            row.append(img_data)
            # manually iterate file index
            file_counter += 1

        # stack each row horizontally in outer loop
        # edge case: one column
        if dim[1] == 1:
            build_row = row[0]
        else:
            build_row = np.hstack((row[0],row[1]))

        if dim[1] > 2:
            for c in range(2,dim[1]):
                build_row = np.hstack((build_row,row[c]))
            completed_rows.append(build_row)

    # reset matplotlib s.t. plots are shown
    mpl.use(backend_) # Reset backend
    
    # clean up intermediate files
    shutil.rmtree('intermediate_temp')

    # stack rows to build final image
    # edge case: one row
    print(np.shape(completed_rows[0]))
    print(np.shape(completed_rows[1]))
    if dim[0] == 1:
        final_image = completed_rows[0]
    else:
        final_image = np.vstack((completed_rows[0],completed_rows[1]))

    if dim[0] > 2:
        for r in range(2,dim[0]):
            final_image = np.vstack((final_image,completed_rows[r]))

    # plot image and save
    plt.figure(figsize=(2*dim[1],2*dim[0]),dpi=300)
    plt.imshow(final_image)
    plt.axis('off')
    plt.savefig(save_name,bbox_inches='tight')
    plt.show()

	

#############################
# Compare Priors to Eachother
#############################
def plot_compare_priors(params_list,priors_dicts,title):
    """
    Args:
        params_list (list): list of strings of param names to be plotted
        priors_dict ([dictionaries]): dictionaries contain a scipy.stats random 
            variable for each parameter 
            (key = param name, value = scipy.stats rv)
            and a label for plot legend (key='label', value = string)
        title (string): title name for plot, and name of file saved
    """

    supported_params = {'theta_E','gamma','q','phi','gamma_ext','phi_ext',
        'source_redshift','lens_redshift','lens_n_sersic','lens_R_sersic',
        'lens_q','lens_phi','lens_magnitude','source_n_sersic',
        'source_R_sersic','point_source_magnitude'}

    # TODO: subplots
    num_r = 2
    num_c = int(np.ceil(np.size(params_list)/2))
    fig, ax = plt.subplots(num_r, num_c,figsize=(10*num_c,7.5*num_r))
    p_index = 0

    for r in range(0,num_r):
        for c in range(0,num_c):

            if num_c ==1:
                axis = ax[r]
            else:
                axis = ax[r,c]
            
            # skip last box for odd-number params
            if p_index >= np.size(params_list):
                axis.axis('off')
                continue
            
            p = params_list[p_index]

            # skip params not in .csv log
            if p not in supported_params:
                # skip to next iteration
                print("Skipping parameter: ", p)
                print("Not supported by plot_compare_prior()")
                axis.axis('off')
                continue

            # use .rvs() samples to define x-range of plot
            rv = []
            rv_samples = []
            for dict in priors_dicts:
                dist = dict[p]
                rv.append(dist)
                samples = dist.rvs(100)
                rv_samples.append(samples)

            # TODO: Debug this
            x_min = np.min(np.min(rv_samples))
            x_max = np.max(np.max(rv_samples))
            x = np.linspace(x_min-0.1,x_max+0.1,100)

            # Plot priors
            for i,dist in enumerate(rv):
                axis.plot(x,dist.pdf(x),label=priors_dicts[i]['label'])

            axis.get_yaxis().set_ticks([])

            # label axes
            axis.set_title(p,fontsize=15)
            axis.legend(fontsize=12)

            p_index += 1

    plt.subplots_adjust(wspace=0.2,hspace=0.4)
    plt.suptitle(title)
    plt.savefig(title+'.png',bbox_inches='tight')
    plt.show()


###########################
# Compare Dataset to Prior
###########################
def plot_compare_prior_dataset(params_list,priors_dict,title,df=None,
                       df_suffix='_STRIDES_median',special_set=None):
    """
    args:
        params_list (list): list of strings of param names to be plotted
        priors_dict (dictionary): contains a scipy.stats random variable for 
            each parameter (key = param name, value = scipy.stats rv)
        title (string): title name for plot, and name of file saved
        df (pandas dataframe): all info from lens catalog .csv. If None, does
            not scatter parameter values on top of pdfs
        df_suffix (string): suffix that defines which parameter values to use 
            from the .csv (i.e. from forward modeling, BNN results, etc.)
        special_set (list): list of strings of system names that will be 
            highlighted on scatter plot
    """

    supported_params = {'theta_E','gamma','q','phi','gamma_ext','phi_ext',
        'source_redshift','lens_redshift','lens_n_sersic','lens_R_sersic',
        'lens_q','lens_phi','lens_magnitude','source_n_sersic',
        'source_R_sersic','point_source_magnitude'}

    # TODO: subplots
    num_r = 2
    num_c = int(np.ceil(np.size(params_list)/2))
    fig, ax = plt.subplots(num_r, num_c,figsize=(6*num_c,4*num_r))
    p_index = 0

    for r in range(0,num_r):
        for c in range(0,num_c):
            
            # skip last box for odd-number params
            if p_index >= np.size(params_list):
                ax[r,c].axis('off')
                continue
            
            p = params_list[p_index]

            # skip params not in .csv log
            if p not in supported_params:
                # skip to next iteration
                print("Skipping parameter: ", p)
                print("Not supported by plot_compare_prior()")
                ax[r,c].axis('off')
                continue

            # use .rvs() samples to define x-range of plot
            rv = priors_dict[p]
            rv_samples = rv.rvs(100)

            # load in parameter values to scatter 
            if df is not None:
                if p in {'source_redshift','lens_redshift'} :
                    param_vals = df[p].to_numpy()

                # special case for lens magnitude
                elif p == 'lens_magnitude':
                    amplitudes = df['lens_SB_STRIDES_median'].to_numpy()
                    param_vals = []
                    for i in range(0,np.size(amplitudes)):
                        # ellipticity translation
                        q = df['lens_q_STRIDES_median'][i]
                        phi = df['lens_phi_STRIDES_median'][i]
                        e1 = (1 - q)/(1+q) * np.cos(2*phi)
                        e2 = (1 - q)/(1+q) * np.sin(2*phi)
                        # lenstronomy key word arguments for SERSIC_ELLIPSE
                        kwa = {'amp': amplitudes[i], 
                            'R_sersic': df['lens_R_sersic_STRIDES_median'][i], 
                            'n_sersic': df['lens_n_sersic_STRIDES_median'][i], 
                            'e1': e1, 
                            'e2': e2, 
                            'center_x': 0, 
                            'center_y': 0}
                        # amp 2 mag conversion
                        mag = amp_2_apparent_mag('SERSIC_ELLIPSE',kwa,
                            df['AB_zeropoint'][i])
                        param_vals.append(mag)
                    # final array of magnitudes
                    param_vals = np.asarray(param_vals)

                # special case for point source magnitude
                elif p == 'point_source_magnitude':
                    param_vals = []
                    for i in range(0,len(df['name'])):

                        # skip case where lens not fitted by STRIDES
                        if df['lens_R_sersic_STRIDES_median'][i] == np.nan:
                            continue

                        # load in image magnitudes & magnifications, AB zeropoint
                        ps_magnitudes = [df['image_A_magnitude_STRIDES_median'][i],
                                        df['image_B_magnitude_STRIDES_median'][i],
                                        df['image_C_magnitude_STRIDES_median'][i],
                                        df['image_D_magnitude_STRIDES_median'][i]]
                        ps_magnifications = [df['image_A_magnif_STRIDES_median'][i],
                                        df['image_B_magnif_STRIDES_median'][i],
                                        df['image_C_magnif_STRIDES_median'][i],
                                        df['image_D_magnif_STRIDES_median'][i]]
                        ab_zeropoint = df['AB_zeropoint'][i]

                        # convert to source apparent magnitude
                        source_mag = ps_image2source_magnitude(ps_magnitudes,
                                        ps_magnifications,ab_zeropoint)
                        param_vals.append(source_mag)

                    # final array of magnitudes
                    param_vals = np.asarray(param_vals)
            
                # all other parameters
                else:
                    param_vals = df[p+df_suffix].to_numpy()

                # get rid of nans from lens w/out fit results
                param_vals = param_vals[~np.isnan(param_vals)]

                # if q or q_ext, must transform from degrees to radians
                if p in {'phi','phi_ext','lens_phi'}:
                    param_vals = np.radians(param_vals)

            # plot prior pdf
            if df is None:
                x_min = np.min(rv_samples)
                x_max = np.max(rv_samples)
            else:
                x_min = np.min((np.min(rv_samples),np.min(param_vals)))
                x_max = np.max((np.max(rv_samples),np.max(param_vals)))
            x = np.linspace(x_min-0.1,x_max+0.1,100)
            ax[r,c].plot(x,rv.pdf(x),label='Prior')

            # highlight 2-component lens light points for lens magnitude
            if p == 'lens_magnitude' and df is not None:
                n_sersic = df['lens_n_sersic_STRIDES_median'].to_numpy()
                n_sersic = n_sersic[~np.isnan(n_sersic)]
                ax[r,c].scatter(param_vals[n_sersic==4.0],
                                rv.pdf(param_vals[n_sersic==4.0]),
                                color='orange',s=30,
                                label='2-Component Lens')
                ax[r,c].scatter(param_vals[n_sersic!=4.0],
                                rv.pdf(param_vals[n_sersic!=4.0]),
                                color='red',s=30,label='Single Component Lens')

            # normal scatter for other parameters
            elif df is not None:
                ax[r,c].scatter(param_vals,rv.pdf(param_vals),color='red',
                                s=30,label='STRIDES')

            # highlight special subset on all scatters
            if special_set is not None:
                names = df['name'].to_numpy()
                names = names[~np.isnan(
                              df['lens_SB_STRIDES_median'].to_numpy())]
                first_name = True
                for name in special_set:
                    if first_name:
                        ax[r,c].scatter(param_vals[names==name],
                                    rv.pdf(param_vals[names==name]),color='limegreen',
                                    s=100,marker='*',label='Subset')
                        first_name = False
                    else:
                        ax[r,c].scatter(param_vals[names==name],
                                    rv.pdf(param_vals[names==name]),color='limegreen',
                                    s=100,marker='*')

            ax[r,c].get_yaxis().set_ticks([])

            # label axes
            ax[r,c].set_title(p,fontsize=10)
            if p_index == 0 or p == 'lens_magnitude':
                ax[r,c].legend(fontsize=7)

            p_index += 1

    plt.subplots_adjust(wspace=0.2,hspace=0.4)
    plt.suptitle(title)
    plt.savefig(title+'.png',bbox_inches='tight')
    plt.show()


def prior_corner_plot(prior_dict,N):
    """
    Args:
        prior_dict (dictionary): Dictionary containing callable scipy.stats 
            objects for each parameter
        N (int): number of samples
    """
    gamma_samples = prior_dict['main_deflector']['gamma_lens'](size=N)
    theta_E_samples = prior_dict['main_deflector']['theta_E'](size=N)
    q_samples = prior_dict['main_deflector']['q'](size=N)
    phi_samples = prior_dict['main_deflector']['phi'](size=N)
    gamma_ext_samples = prior_dict['main_deflector']['gamma_ext'](size=N)
    phi_ext_samples = prior_dict['main_deflector']['phi_ext'](size=N)
    data = np.hstack((gamma_samples.reshape([N,1]),theta_E_samples.reshape([N,1]),
        q_samples.reshape([N,1]),phi_samples.reshape([N,1]),
        gamma_ext_samples.reshape([N,1]),phi_ext_samples.reshape([N,1])))
    corner.corner(data,labels=['$\gamma_{lens}$',
        '\u03B8$_E$','$q_{lens}$','$\phi_{lens}$','$\gamma_{ext}$','$\phi_{ext}$'],
        color='tab:grey',fill_contours=True,alpha=0.5,smooth=1.5,
        plot_datapoints=False,levels=[0.68, 0.95])
    plt.savefig('corner_plot.png')

def retrieve_paltas_samples_slow(paltas_config_file,component_param_dict,
    num_params,num_samples):
    """
    Loops through sampling method used by paltas and returns samples for
    chosen parameters

    Args:
        paltas_config_file (string): path to paltas config file
        component_param_dict (dict): dictionary with keys that specify a
            paltas component, and values that are a list of params 
            i.e. dict = {
                'main_deflector_parameters':['gamma','theta_E',...],
                'source_parameters':['n_sersic',...],
                ...
            }
        num_params (int): number parameters (needed to allocate array)
        num_samples (int): number desired samples
    """

    config_handler = ConfigHandler(paltas_config_file)

    samples = np.empty((num_samples,num_params))
    for i in range(0,num_samples):
        row = []
        config_handler.draw_new_sample()
        sample = config_handler.get_current_sample()

        for k in component_param_dict.keys():
             for param in component_param_dict[k]:
                  row.append(sample[k][param])

        samples[i] = np.asarray(row)


    return samples


def prior_corner_plot_slow(paltas_config_file,N,doppel_files=None):
    """
    Slower verion, loops thru sampling method used by paltas

    Args:
        paltas_config_file (string): Paltas config file
        N (int): number of samples
        doppel_files [string]: list of doppelganger paltas config files
    """
    config_handler = ConfigHandler(paltas_config_file)
    cosmology_params = {
			'cosmology_name': 'planck18'
		}
    cosmo = get_cosmology(cosmology_params)

    ######################
    # Lens Mass Parameters
    ######################
    num_params = 7
    data_lm = np.empty((N,num_params))
    for i in range(0,N):
        row = []
        config_handler.draw_new_sample()
        sample = config_handler.get_current_sample()
        row.append(sample['main_deflector_parameters']['gamma'])
        row.append(sample['main_deflector_parameters']['theta_E'])
        row.append(sample['main_deflector_parameters']['e1'])
        row.append(sample['main_deflector_parameters']['e2'])
        row.append(sample['main_deflector_parameters']['gamma1'])
        row.append(sample['main_deflector_parameters']['gamma2'])
        row.append(sample['main_deflector_parameters']['z_lens'])
        data_lm[i] = np.asarray(row)

    corner.corner(data_lm,labels=['$\gamma_{lens}$',
        '\u03B8$_E$ (")','$e1$','$e2$','$\gamma1$','$\gamma2$','$z_{lens}$'],
        label_kwargs={'fontsize': 20},color='tab:grey',fill_contours=True,
        alpha=0.5,smooth=1.5,plot_datapoints=False,levels=[0.68, 0.95])
    plt.suptitle('Lens Parameters Prior',fontsize=20)


    ####################
    # Source Parameters
    ####################
    num_params = 4
    data_b = np.empty((N,num_params))
    for i in range(0,N):
        row = []
        config_handler.draw_new_sample()
        sample = config_handler.get_current_sample()
        row.append(sample['lens_light_parameters']['R_sersic'])
        row.append(sample['lens_light_parameters']['n_sersic'])
        row.append(sample['source_parameters']['R_sersic'])
        row.append(sample['source_parameters']['n_sersic'])
        data_b[i] = np.asarray(row)

    figure = corner.corner(data_b,labels=['$R$_$sersic_{lens}$',
        '$n$_$sersic_{lens}$','$R$_$sersic_{src}$', '$n$_$sersic_{src}$'],
        label_kwargs={'fontsize': 20},color='tab:grey',fill_contours=True,
        alpha=0.5,smooth=1.5,
        plot_datapoints=False,levels=[0.68, 0.95])
    plt.suptitle('Sersic Prior',fontsize=20)

    axes_sersic = np.array(figure.axes).reshape((num_params, num_params))


    #######################
    # Magnitude Parameters
    #######################
    num_params = 3
    data_b = np.empty((N,num_params))
    for i in range(0,N):
        row = []
        config_handler.draw_new_sample()
        sample = config_handler.get_current_sample()
        row.append(sample['lens_light_parameters']['mag_app'])
        row.append(sample['source_parameters']['mag_app'])
        row.append(sample['point_source_parameters']['mag_app'])
        data_b[i] = np.asarray(row)

    figure = corner.corner(data_b,labels=['$app$_$mag_{lens}$',
        '$app$_$mag_{src}$','$app$_$mag_{ps}$'],label_kwargs={'fontsize': 20},
        color='tab:grey',fill_contours=True,alpha=0.5,smooth=1.5,
        plot_datapoints=False,levels=[0.68, 0.95])
    plt.suptitle('Magnitudes Prior',fontsize=20)
    
    axes = np.array(figure.axes).reshape((num_params, num_params))

	#######################
	# Source / Lens Center
    #######################
    num_params = 4
    data_b = np.empty((N,num_params))
    for i in range(0,N):
        row = []
        config_handler.draw_new_sample()
        sample = config_handler.get_current_sample()
        row.append(sample['main_deflector_parameters']['center_x'])
        row.append(sample['main_deflector_parameters']['center_y'])
        row.append(sample['source_parameters']['center_x'])
        row.append(sample['source_parameters']['center_y'])
        data_b[i] = np.asarray(row)

    figure = corner.corner(data_b,labels=['$x_{lens}$',
        '$y_{lens}$','$x_{src}$','$y_{src}$'],label_kwargs={'fontsize': 20},
        color='tab:grey',fill_contours=True,alpha=0.5,smooth=1.5,
        plot_datapoints=False,levels=[0.68, 0.95])
    plt.suptitle('Center Coordinates Prior',fontsize=20)
    
    axes_coord = np.array(figure.axes).reshape((num_params, num_params))


    if doppel_files is None:
        return

    for f in doppel_files:
        ch = ConfigHandler(f)
        ch.draw_new_sample()
        sample = ch.get_current_sample()
        axes[1,0].scatter(sample['lens_light_parameters']['mag_app'],
            sample['source_parameters']['mag_app'],color='b')
        axes[2,0].scatter(sample['lens_light_parameters']['mag_app'],
            sample['point_source_parameters']['mag_app'],color='b')
        axes[2,1].scatter(sample['source_parameters']['mag_app'],
            sample['point_source_parameters']['mag_app'],color='b')
        
        axes_sersic[1,0].scatter(sample['lens_light_parameters']['R_sersic'],
            sample['lens_light_parameters']['n_sersic'],color='b')
        axes_sersic[2,0].scatter(sample['lens_light_parameters']['R_sersic'],
            sample['source_parameters']['R_sersic'],color='b')
        axes_sersic[2,1].scatter(sample['lens_light_parameters']['n_sersic'],
            sample['source_parameters']['R_sersic'],color='b')
        axes_sersic[3,0].scatter(sample['lens_light_parameters']['R_sersic'],
            sample['source_parameters']['n_sersic'],color='b')
        axes_sersic[3,1].scatter(sample['lens_light_parameters']['n_sersic'],
            sample['source_parameters']['n_sersic'],color='b')
        axes_sersic[3,2].scatter(sample['source_parameters']['R_sersic'],
            sample['source_parameters']['n_sersic'],color='b')
        


def plot_coverage_plain(y_pred,y_test,std_pred,parameter_names,
                        fontsize=20,show_error_bars=True,n_rows=4):
    """ Generate plots for the 1D coverage of each parameter.

    Args:
        y_pred (np.array): A (batch_size,num_params) array containing the
            mean prediction for each Gaussian
        y_test (np.array): A (batch_size,num_params) array containing the
            true value of the parameter on the test set.
        std_pred (np.array): A (batch_size,num_params) array containing the
            predicted standard deviation for each parameter.
        parameter_names ([str,...]): A list of the parameter names to be
            printed in the plots.ed.
        color_map ([str,...]): A list of at least 4 colors that will be used
            for plotting the different coverage probabilities.
        block (bool): If true, block excecution after plt.show() command.
        fontsize (int): The fontsize to use for the parameter names.
        show_error_bars (bool): If true plot the error bars on the coverage
            plot.
        n_rows (int): The number of rows to include in the subplot.
    """
    num_params = len(parameter_names)
    error = y_pred - y_test
    # Define the covariance masks for our coverage plots.
    plt.rcParams['figure.dpi'] = 200
    fig = plt.figure(figsize=(20,16),num=1)
    plt.rcParams['figure.dpi'] = 80
    fig.subplots_adjust(hspace=0.3,wspace=0.3,top=0.92)
    custom_color = colors.to_hex('slateblue')
    #custom_color_hex = scale_hex_color(custom_color,0.9)
    for i in range(len(parameter_names)):
        plt.subplot(n_rows, int(np.ceil(num_params/n_rows)), i+1)
        plt.scatter(y_test[:,i],y_pred[:,i],s=60,color=custom_color)

        # Include the correlation coefficient squared value in the plot.
        straight = np.linspace(np.min(y_test[:,i]),np.max(y_test[:,i]),10)
        plt.plot(straight, straight,color='k',zorder=100)
        plt.title(parameter_names[i],fontsize=fontsize)
        plt.ylabel('Prediction',fontsize=fontsize)
        plt.xlabel('True Value',fontsize=fontsize)

        #Add mean error/MAE/P
        ax = plt.gca()
        #mean_error = np.mean(error[:,i])
        #plt.text(0.6,0.04,
        #	'Mean Error: %.3f'%(mean_error),{'fontsize':fontsize},transform=ax.transAxes)
        MAE = np.median(np.abs(error[:,i]))
        plt.text(0.73,0.05,
            'MAE: %.3f'%(MAE),{'fontsize':15},transform=ax.transAxes)
        P = np.median(std_pred[:,i])
        plt.text(0.73,0.13,'P: %.3f'%(P),{'fontsize':15},transform=ax.transAxes)


# taken from paltas: https://github.com/swagnercarena/paltas/blob/main/paltas/Analysis/posterior_functions.py
def plot_coverage(y_pred,y_test,std_pred,parameter_names,
	fontsize=15,n_rows=4,bin_min=0.9,bin_max=250,title=None):
	""" Generate plots for the 1D coverage of each parameter.

	Args:
		y_pred (np.array): A (batch_size,num_params) array containing the
			mean prediction for each Gaussian
		y_test (np.array): A (batch_size,num_params) array containing the
			true value of the parameter on the test set.
		std_pred (np.array): A (batch_size,num_params) array containing the
			predicted standard deviation for each parameter.
		parameter_names ([str,...]): A list of the parameter names to be
			printed in the plots.ed.
		fontsize (int): The fontsize to use for the parameter names.
		n_rows (int): The number of rows to include in the subplot.
        bin_min (float): Min value for greyscale colorbar
        bin_max (float): Max value for greyscale colorbar
	"""
	num_params = len(parameter_names)
	error = y_pred - y_test
	std_devs_away = np.abs(error)/std_pred
    # custom greyscale colormap
	from matplotlib import cm
	from matplotlib import colors
	min_val, max_val = 0.1,1.0
	n = 100
	orig_cmap = cm.Greys
	my_colors = orig_cmap(np.linspace(min_val, max_val, n))
	cmap = colors.LinearSegmentedColormap.from_list("mycmap", my_colors)
    # plotting
	for i in range(len(parameter_names)):
		plt.subplot(n_rows+1, int(np.ceil(num_params/n_rows)), i+1)
		#plt.scatter(y_test[:,i],y_pred[:,i],c=std_devs_away[:,i],cmap=cmap,vmin=0,vmax=3,
		#	s=5,zorder=100)
		#plt.colorbar()

		# contour 1sigma,2sigma,3sigma for test set

		# reduce y_test, y_pred to 3 dec. places
		x = y_test[:,i]
		y = y_pred[:,i]
		# make meshgrid X,Y 3 dec. places
		x_max = np.max([np.max(x),np.max(y)])
		x_min = np.min([np.min(x),np.min(y)])
		X,Y = np.meshgrid(np.linspace(x_min,x_max,10),np.linspace(x_min,x_max,10))
		dx = np.abs(X[0,0] - X[0,1])
		# construct Z: np.zeros(), fill in value where it exists
		Z = np.ones(np.shape(X))*np.nan
		for j in range(0,len(x)):
			x_idx = np.where(np.abs(X[0,:]-x[j])<dx)[0][0]
			y_idx = np.where(np.abs(Y[:,0]-y[j])<dx)[0][0]
			Z[x_idx,y_idx] = std_devs_away[j,i]
		#from skimage.measure import block_reduce
		#Z_interpol = block_reduce(Z,block_size=(2,2), func=np.mean)
		#plt.hist2d(y_test[:,i],y_pred[:,i],cmap='Greys')
		import seaborn
		from matplotlib.colors import LogNorm
		seaborn.histplot(x=y_test[:,i],y=error[:,i],bins=50,cmap='Greys',
			vmin=bin_min,vmax=bin_max)#,norm=LogNorm())
		#seaborn.heatmap(np.flipud(Z),cmap=cmap,vmin=0.0,vmax=3.0,xticklabels=X[0,:],yticklabels=Y[:,0])
		#seaborn.kdeplot(y_test[:,i],y_pred[:,i],color='black',levels=10)
		#plt.imshow(Z)
		#plt.colorbar()
		#plt.contour(block_reduce(X,block_size=(2,2), func=np.mean),
		#	block_reduce(Y,block_size=(2,2), func=np.mean),
		#	Z_interpol,levels=[1.,2.,3.])
		#plt.contour(X,Y,Z,levels=[1.,2.,3.])

		# y=x line & labels
		straight = np.linspace(np.min(y_test[:,i]),np.max(y_test[:,i]),10)
		plt.plot(straight, np.zeros(np.shape(straight)), label='',color='k')
		plt.title(parameter_names[i],fontsize=fontsize+3)
		plt.ylabel('Pred. - Truth',fontsize=fontsize-1)
		plt.xlabel('Truth',fontsize=fontsize-1)
		#plt.legend(**{'fontsize':fontsize},loc=2)

	if title is not None:
		plt.suptitle(title,fontsize=fontsize)

	plt.figure()
	X,Y = np.meshgrid(np.linspace(250.0,0.0,num=100),np.linspace(250.0,0.0,num=100))
	plt.imshow(Y[:,0:10],cmap='Greys',vmin=bin_min,vmax=bin_max)#,norm=LogNorm())
	cbar = plt.colorbar()
	cbar.ax.get_yaxis().labelpad = 8
	cbar.ax.set_ylabel('$N_{val}$', rotation=0, fontsize=12)

def plot_coverage_compare_sequential_reweighted(y_pred_doppel,y_test_doppel,std_pred_doppel,
    y_pred_seq=None, std_pred_seq=None,
    y_pred_reweighted=None,std_pred_reweighted=None,
    learning_params_print=None,bin_max=250,doppelgangers=None,
    title=None,fig_dim=(20,15),x_lim=None,y_lim=None,save_path=None):
    """
    Args:
        y_pred_doppel: test set means
        y_test_doppel: test set ground truth
        std_pred_doppel: test set standard deviations
        y_pred_seq: test set means after sequential
        std_pred_seq: test set std devs after sequential
        y_pred_reweighted: test set means after reweighting
        std_pred_reweighted: test set standard deviations after reweighting
        learning_params_print: names/labels of each parameter in (y_pred)
        bin_max: maximum number in colorscale for validation set scatter
        doppelgangers (bool): whether to plot letters on top of test set to 
            identify dopppelgangers
        title (string): if not None, plt.suptitle() argument
        fig_dim (tuple): figure size
        x_lim (list[tuple]): x lims for each param 
        y_lim (list[tuple]): y lims for each param
    """

    plt.rcParams['figure.dpi'] = 200
    fig = plt.figure(figsize=fig_dim,num=1)
    plt.rcParams['figure.dpi'] = 80
    fig.subplots_adjust(hspace=0.3,wspace=0.3,top=0.92)

    # set up our axes
    n_rows = 3
    for k in range(0,len(learning_params_print)):
        plt.subplot(n_rows+1, int(np.ceil(len(learning_params_print)/n_rows)), k+1)

        straight = np.linspace(np.min(y_test_doppel[:,k]),np.max(y_test_doppel[:,k]),10)
        plt.plot(straight, straight, label='',color='grey')
        plt.title(learning_params_print[k],fontsize=18)
        plt.ylabel('Prediction',fontsize=15-1)
        plt.xlabel('Truth',fontsize=15-1)

    # now, scatter test set error before/after reweighting
    axes = np.array(fig.axes)
    markers = ['$A$','$B$','$C$','$D$','$E$','$F$','$G$','$H$','$I$','$J$','$K$']
    error_doppel = y_pred_doppel - y_test_doppel
    if y_pred_seq is not None:
        error_seq = y_pred_seq - y_test_doppel
    if y_pred_reweighted is not None:
        error_reweighted = y_pred_reweighted - y_test_doppel
    # scatter un-weighted & re-weighted
    for j in range(0,len(y_pred_doppel[:,0])):
        for i in range(0,len(learning_params_print)):
            # scatter initial NPE estimates
            axes[i].scatter(y_test_doppel[j][i],y_pred_doppel[j][i],zorder=100,
                s=60,marker='o',c='slateblue',label='Initial NPE')
            # scatter sequential NPE estimates
            if y_pred_seq is not None:
                axes[i].scatter(y_test_doppel[j][i],y_pred_seq[j][i],zorder=100,
                    s=60,marker='o',c='mediumseagreen',label='Sequential NPE')
            # scatter reweighted NPE estimates
            if y_pred_reweighted is not None:
                axes[i].scatter(y_test_doppel[j][i],y_pred_reweighted[j][i],zorder=100,
                    s=60,marker='o',c='mediumorchid',label='Reweighted')
            # plot line between the two
            if y_pred_seq is not None:
                axes[i].plot([y_test_doppel[j][i],y_test_doppel[j][i]],
                    [y_pred_doppel[j][i],y_pred_seq[j][i]],color='black')
            #axes[i].plot([y_test_doppel[j][i],y_test_doppel[j][i]],
            #	[y_pred_seq[j][i],y_pred_reweighted[j][i]],color='black')
            # letters on top of scatter if requested
            if doppelgangers is not None:
                axes[i].text(y_test_doppel[j][i],y_pred_doppel[j][i],markers[j],zorder=1000,
                    horizontalalignment='center',verticalalignment='center',fontsize=15)
                axes[i].text(y_test_doppel[j][i],y_pred_reweighted[j][i],
                    markers[j],zorder=1000,horizontalalignment='center',
                    verticalalignment='center',fontsize=15)
            if y_lim is not None:
                axes[i].set_ylim(y_lim[i])
            if x_lim is not None:
                axes[i].set_xlim(x_lim[i])
                        
            if j == 0 and i ==0:
                axes[0].legend(fontsize=15)

    plt.figure(num=1)
    if save_path is not None:
        print('saving figure')
        plt.savefig(save_path,bbox_inches='tight')

    plt.show()

    # plot separate figure with "key" for doppelganger letters
    if doppelgangers is not None:
        plt.figure(figsize=(3,5))
        for m in range(0,len(markers)):
            plt.scatter(1,11-m,marker='o',s=150,facecolors='none', edgecolors='k')
            rvrs = len(markers)-1-m
            plt.text(1,m+1,markers[rvrs],zorder=1000,
                    horizontalalignment='center',verticalalignment='center')
            plt.text(1.05,m+1,doppelgangers[rvrs],verticalalignment='center')
        plt.xlim([.98,1.3])
        plt.ylim([0,len(markers)+1])
   

def plot_coverage_doppels(pred_x_axis,pred_y_axis,std_pred_x_axis=None,
    std_pred_y_axis=None,color='grey',axis_labels=['',''],save_path=None,
    doppelgangers=False):

    # TODO: fix, hardcoded!!
    param_labels =  [r'$\theta_\mathrm{E}$',r'$\gamma_1$',r'$\gamma_2$',r'$\gamma_\mathrm{lens}$',r'$e_1$',
                                    r'$e_2$']
    lens_names_list = ['ATLAS J2344-3056', 'DES J0405-3308','DES J0420-4037','DES J0530-3730','J0029-3814', 'J1131-4419',
    'J2205-3727','PS J1606-2333', 'SDSS J0248+1913', 'SDSS J1251+2935',
    'W2M J1042+1641', 'WG0214-2105', 'WISE J0259-1635']
    # only 11 doppelgangers
    if doppelgangers:
        lens_names_list = ['ATLAS J2344-3056', 'DES J0405-3308','DES J0420-4037','J1131-4419', 'J2145+6345',
'J2205-3727','PS J1606-2333', 'SDSS J0248+1913', 
'W2M J1042+1641', 'WG0214-2105', 'WISE J0259-1635']
    markers = ['$A$','$B$','$C$','$D$','$E$','$F$','$G$','$H$','$I$','$J$','$K$','$L$','$M$']

    fig, axes = plt.subplots(2,4,figsize=(26,12))
    for l in range(0,np.shape(pred_x_axis)[0]):
        for p in range(0,6):

            if l == 0:
                ultra_min = np.min(np.asarray([np.min(pred_x_axis[:,p]),np.min(pred_y_axis[:,p])]))
                ultra_max = np.max(np.asarray([np.max(pred_x_axis[:,p]),np.max(pred_y_axis[:,p])]))
                axes[p//3,p%3].plot(np.arange(ultra_min,ultra_max,0.01),np.arange(ultra_min,ultra_max,0.01),color='grey')

                axes[p//3,p%3].text(0.75,0.05,
                    r'$\rho$ = %.2f'%(np.corrcoef(pred_x_axis[:,p],pred_y_axis[:,p])[0,1]),
                    transform=axes[p//3,p%3].transAxes,fontsize=15)

            yerr = None
            xerr = None
            if std_pred_x_axis is not None:
                 xerr = std_pred_x_axis[l,p]
            if std_pred_y_axis is not None:
                 yerr = std_pred_y_axis[l,p]
            axes[p//3,p%3].errorbar(x=pred_x_axis[l,p],y=pred_y_axis[l,p],
                xerr=xerr,yerr=yerr,c=color)
            
            axes[p//3,p%3].scatter(pred_x_axis[l,p],pred_y_axis[l,p],s=240,c=color,zorder=500)
            axes[p//3,p%3].text(pred_x_axis[l,p],pred_y_axis[l,p],markers[l],zorder=1000,
                        horizontalalignment='center',verticalalignment='center',fontsize=14)
 
            if l == 0:
                axes[p//3,p%3].set_xlabel(axis_labels[0])
                axes[p//3,p%3].set_ylabel(axis_labels[1])
                axes[p//3,p%3].set_title(param_labels[p],fontsize=24)    

    for m in range(0,len(lens_names_list)):
        axes[0,3].scatter(1,len(lens_names_list)-m,marker='o',s=240,
            facecolors='none', edgecolors='k')
        rvrs = m
        axes[0,3].text(1,len(lens_names_list)-m,markers[rvrs],zorder=1000,
                horizontalalignment='center',verticalalignment='center',fontsize=13)
        axes[0,3].text(1.02,len(lens_names_list)-m,lens_names_list[rvrs],verticalalignment='center',fontsize=15)

    axes[0,3].set_xlim([.975,1.4])
    axes[0,3].set_ylim([0,len(markers)+1])
    axes[0,3].set_xticks([])
    axes[0,3].set_yticks([])
    axes[0,3].axis('off')
    axes[1,3].axis('off')

    if save_path is not None:
        plt.savefig(save_path)

	
def table_metrics_compare_list(y_pred_list,std_pred_list,y_test,outfile,list_labels,doppel=False):
    """Same concept as table_metrics, but compares a list of results
	Args:
	"""
    for i in range(0,len(y_pred_list)):
        
        print(list_labels[i])
        table_metrics(y_pred_list[i],y_test,std_pred_list[0],None)
        print(" ")
         


def table_metrics(y_pred,y_test,std_pred,outfile,doppel=False):
	"""
	Args:
		y_pred (np.array): A (batch_size,num_params) array containing the
			mean prediction for each Gaussian
		y_test (np.array): A (batch_size,num_params) array containing the
			true value of the parameter on the test set.
		std_pred (np.array): A (batch_size,num_params) array containing the
			predicted standard deviation for each parameter.
		outfile (string): File to write latex source to
	"""
	error = y_pred - y_test
	#Add mean error/MAE/P
	mean_error = np.mean(error,axis=0)
	median_error = np.median(error,axis=0)
    # mean bias in number of std deviations
	mean_bias_std_devs = np.mean(error/std_pred,axis=0)
	median_bias_std_devs = np.median(error/std_pred,axis=0)
	#plt.text(0.73,0.05,
	#	'Mean Error: %.3f'%(mean_error),{'fontsize':fontsize},transform=ax.transAxes)
	MAE = np.median(np.abs(error),axis=0)
	mean_AE = np.mean(np.abs(error),axis=0)
	avg_percent_error = np.mean(np.abs(error)/y_test,axis=0)*100
	#plt.text(0.73,0.11,
	#	'MAE: %.3f'%(MAE),{'fontsize':fontsize},transform=ax.transAxes)
	P = np.median(std_pred,axis=0)
	#plt.text(0.73,0.16,'P: %.3f'%(P),{'fontsize':fontsize},transform=ax.transAxes)
	
    # how to do % precision?
	avg_percent_precision = np.mean(std_pred/np.abs(y_pred),axis=0)*100
	median_percent_precision = np.median(std_pred/np.abs(y_pred),axis=0)*100

	cov_masks = [np.abs(error)<=std_pred,np.abs(error)<2*std_pred,
		np.abs(error)<3*std_pred]

	#metrics = [mean_bias_std_devs,median_bias_std_devs,MAE,P,
    #    avg_percent_precision,median_percent_precision]
	metrics = [median_bias_std_devs,median_error,MAE,P,avg_percent_error,avg_percent_precision]
		
	if outfile is not None:
		f = open(outfile,'w')
	else:
		f = sys.stdout

	f.write('\hline')
	f.write('\n')
	for i,lab in enumerate(['Median Bias (in $\sigma$)','Median Error',
        'MAE','Median($\sigma$)','Avg. Error','Avg Prec.']):
		f.write(lab)
		f.write(' ')

		for m in metrics[i]:
			f.write('& ')
			f.write(str(np.around(m,2)))
			f.write(' ')


		f.write(r'\\')
		f.write('\n')
		f.write('\hline')
		f.write('\n')

	"""
	# write % contained in 1,2,3 sigma
	for j,cov_mask in enumerate(cov_masks):
		f.write(r'\% contained '+str(j+1)+'$\sigma$')
		f.write(' ')

		for k in range(0,len(metrics[0])):
			f.write('& ')
			if doppel:
				f.write(str(np.sum(cov_mask[:,k])) + '/' + str(len(error)))
			else:
				f.write(str(np.around(np.sum(cov_mask[:,k])/len(error),2)))
			f.write(' ')

		f.write(r'\\')
		f.write('\n')
		f.write('\hline')
		f.write('\n')
    """
                        
	if outfile is not None:
		f.close()
			
def overlay_NPE_contours(y_pred_list,prec_pred_list,color_list,param_labels,
    truths=None,truth_color='black',model_labels=None,truth_label=None,
    titles=None,save_prefix=None):
    """
    Args:
        y_pred_list ([]): list of y_pred for each modeling type
            size: (num_modeling_types, num_lenses, num_params)
        prec_pred_list ([]): list of prec_pred for each modeling type
            size: (num_modeling_types, num_lenses, num_params, num_params)
        color_list ([]):
        param_labels
        truths: ground truth for each lens
        truth_color: color for ground truth lines
        titles ([string] or None)
        save_prefix (string or Nones)
    """
     
    y_pred_list = np.asarray(y_pred_list)
    prec_pred_list = np.asarray(prec_pred_list)
    num_params = np.shape(y_pred_list)[2]

    # loop thru lenses
    for l in range(0,np.shape(y_pred_list)[1]):
        figure = plt.figure(figsize=(10,10))
        # loop thru modeling types
        for m in range(0,np.shape(y_pred_list)[0]):
            samps = multivariate_normal(mean=y_pred_list[m,l],
				cov=np.linalg.inv(prec_pred_list[m,l])).rvs(size=int(5e3))
               
            if truths is not None and m==(np.shape(y_pred_list)[0]-1):
                corner.corner(samps,labels=np.asarray(param_labels),bins=20,
                    show_titles=False,plot_datapoints=False,label_kwargs=dict(fontsize=30),
                    levels=[0.68,0.95],color=color_list[m],fill_contours=True,smooth=1.0,
                    hist_kwargs={'density':True,'color':color_list[m],'lw':3},
                    title_fmt='.2f',max_n_ticks=3,fig=figure,
                    truths=truths[l],truth_color=truth_color)
            else:
                corner.corner(samps,labels=np.asarray(param_labels),bins=20,
                    show_titles=False,plot_datapoints=False,label_kwargs=dict(fontsize=30),
                    levels=[0.68,0.95],color=color_list[m],fill_contours=True,smooth=1.0,
                    hist_kwargs={'density':True,'color':color_list[m],'lw':3},
                    title_fmt='.2f',max_n_ticks=3,fig=figure)
            
        axes = np.array(figure.axes).reshape((num_params, num_params))
        custom_lines = []
        for color in color_list:
            custom_lines.append(Line2D([0], [0], color=color, lw=4))
        if truths is not None:
             custom_lines.append(Line2D([0], [0], color=truth_color, lw=4))
             if model_labels is not None:
                model_labels.append(truth_label)
        if model_labels is not None:
            axes[0,num_params-2].legend(custom_lines,model_labels,frameon=False,
                fontsize=30,loc=10)
        
        if titles is not None:
            #plt.suptitle(titles[l],fontsize=20)       
            if save_prefix is not None:
                plt.savefig(save_prefix+titles[l]+'.pdf',bbox_inches = "tight")

def plot_NPE_contours(train_mean,train_scatter,y_pred_doppel,prec_pred_doppel,
					y_test_doppel,titles,proposal_mean=None,
					proposal_scatter=None,truncated=True):

	"""
	Args:
	"""

	cov_pred_doppel = np.linalg.inv(prec_pred_doppel)
	prec_pred_doppel_truncated = np.linalg.inv(cov_pred_doppel[:,:6,:6])

	samps_training_prior = multivariate_normal(mean=train_mean,
		cov=np.diag(np.asarray(train_scatter)**2)).rvs(size=int(5e3))

	for i in range(0,y_pred_doppel.shape[0]):

		if truncated:
			samps_original = multivariate_normal(mean=y_pred_doppel[i,:6],
				cov=np.linalg.inv(prec_pred_doppel_truncated[i,:,:])).rvs(size=int(5e3))
			param_labels = [r'$\theta_\mathrm{E}$',r'$\gamma_1$',r'$\gamma_2$',r'$\gamma_\mathrm{lens}$',r'$e_1$',
								r'$e_2$']
		else:
			samps_original = multivariate_normal(mean=y_pred_doppel[i,:10],
				cov=cov_pred_doppel[i,:10,:10]).rvs(size=int(5e3))
			param_labels = [r'$\theta_\mathrm{E}$',r'$\gamma_1$',r'$\gamma_2$',r'$\gamma_\mathrm{lens}$',r'$e_1$',
								r'$e_2$',r'$x_{lens}$',r'$y_{lens}$',r'$x_{src}$',r'$y_{src}$']
			
		truth_color = 'black'

		#figure = corner.corner(samps_training_prior,labels=np.asarray(param_labels),bins=20,
		#			show_titles=False,plot_datapoints=False,label_kwargs=dict(fontsize=50),
		#			levels=[0.68,0.95],color='grey',fill_contours=True,smooth=1.0,
		#			hist_kwargs={'density':True,'color':'grey','lw':3},
		#			title_fmt='.2f',max_n_ticks=3,fig=None)
			
		if proposal_mean is not None:
               
			samps_proposal = multivariate_normal(mean=proposal_mean[i,:10],
				cov=np.diag(np.asarray(proposal_scatter[i,:10])**2)).rvs(size=int(5e3))
               
			figure = corner.corner(samps_proposal,labels=np.asarray(param_labels),bins=20,
					show_titles=False,plot_datapoints=False,label_kwargs=dict(fontsize=50),
					levels=[0.68,0.95],color='lightsteelblue',fill_contours=True,smooth=1.0,
					hist_kwargs={'density':True,'color':'lightsteelblue','lw':3},
					title_fmt='.2f',max_n_ticks=3,fig=None)

		if truncated:
			doppel_truth = y_test_doppel[i,:6]
			num_params = 6

		else:
			doppel_truth = y_test_doppel[i,:10]
			num_params = 10

		corner.corner(samps_original,labels=np.asarray(param_labels),bins=20,
					show_titles=False,plot_datapoints=False,label_kwargs=dict(fontsize=50),
					levels=[0.68,0.95],color=COLORS['hyperparam'],fill_contours=True,smooth=1.0,
					hist_kwargs={'density':True,'color':COLORS['hyperparam'],'lw':3},title_fmt='.2f',max_n_ticks=3,fig=figure,
					truths=doppel_truth,
					truth_color=truth_color)#,range=np.ones(6)*0.98)

		
		axes = np.array(figure.axes).reshape((num_params, num_params))
		custom_lines = []
		colors_list = ['grey',COLORS['hyperparam']]
		if proposal_mean is not None:
			colors_list = ['grey','lightsteelblue',COLORS['hyperparam']]
		for color in colors_list:
			custom_lines.append(Line2D([0], [0], color=color, lw=4))
		legend_labels = ['Training Prior', 'paltas Posterior']
		if proposal_mean is not None:
			legend_labels = ['Training Prior', 'NPE Proposal', 'paltas Posterior']
		axes[0,num_params-1].legend(custom_lines,legend_labels,frameon=False,
			fontsize=20)

		plt.suptitle(titles[i],fontsize=30)

def early_stopping_epoch(log_file,num_before_stopping=10):
	df = pd.read_csv(log_file)
	val_loss = df['val_loss'].to_numpy()

	min_val_loss = np.inf
	chosen_epoch = np.nan
	num_waited = 0
	for i,v in enumerate(val_loss):
		if v < min_val_loss:
			min_val_loss = v
			chosen_epoch = i+1 
			num_waited = 0
		else: 
			num_waited += 1
						
			if num_waited == num_before_stopping:
				break			
						
	return chosen_epoch

def loss_plots(log_files,training_sizes,labels,square_error=False,
               chosen_epoch=None,xscale=True,stopping_epoch=None,
               colorname="mediumseagreen"):
	"""
	Args:
		log_files [string]: list of .csv files containing training log
		training_sizes [int]: list of training set sizes
		labels [string]: list of legend labels for each file
        square_error (bool): Whether square_error is part of the log
        chosen_epoch (int or None): if not None, plots a star at the location
			of the chosen epoch
	"""
     
	color_hex = colors.cnames[colorname] 
	color_pairs = [
		[scale_hex_color(color_hex, 0.6),scale_hex_color(color_hex, 1.6)] 
	]
		
	fig, ax = plt.subplots(1, 2,figsize=(15,7))
	for i,f in enumerate(log_files):
		df = pd.read_csv(f)
		loss = df['loss'].to_numpy()
		val_loss = df['val_loss'].to_numpy()
		if stopping_epoch:
			loss = loss[:stopping_epoch]
			val_loss = val_loss[:stopping_epoch]
		steps = (np.arange(len(loss))+1) * (training_sizes[i]/512)
		lr = 5e-3 * 0.98 ** (steps / 1e4)
		ax[0].plot(steps,loss,color=color_pairs[i][0],label=labels[i]+' Training Loss')
		ax[0].plot(steps,val_loss,color=color_pairs[i][1],label=labels[i]+' Val. Loss')
		if chosen_epoch is not None:
			ax[0].scatter(steps[chosen_epoch-1],val_loss[chosen_epoch-1],
                 marker='*',color='magenta',s=100,zorder=100)
		ax[0].set_xlabel('Steps')
		ax[0].set_ylabel('Loss')
		if xscale:
			ax[0].set_xscale('log')
		ax[0].legend()
		ax[0].grid()
		ax[1].plot(lr, loss,color=color_pairs[i][0],label=labels[i]+' Training Loss')
		ax[1].plot(lr, val_loss,color=color_pairs[i][1],label=labels[i]+' Val. Loss')
		ax[1].set_xlabel('Learning Rate')
		ax[1].set_ylabel('Loss')
		ax[1].legend()
		if xscale:
			ax[1].set_xscale('log')
		ax[1].grid()
				
		print("Final Learning Rate: ", lr[-1])

	if square_error:
		fig1,ax1 = plt.subplots(1,2,figsize=(15,7))
		for i,f in enumerate(log_files):
			df = pd.read_csv(f)
			loss = df['square_error'].to_numpy()
			val_loss = df['val_square_error'].to_numpy()
			steps = (np.arange(len(loss))+1) * (training_sizes[i]/512)
			lr = 5e-3 * 0.98 ** (steps / 1e4)
			ax1[0].plot(steps,loss,color=color_pairs[i][0],label=labels[i]+' Training SE')
			ax1[0].plot(steps,val_loss,color=color_pairs[i][1],label=labels[i]+' Val. SE')
			ax1[0].set_xlabel('Steps')
			ax1[0].set_ylabel('SE')
			ax1[0].set_xscale('log')
			ax1[0].legend()
			ax1[0].grid()
			ax1[1].plot(lr, loss,color=color_pairs[i][0],label=labels[i]+' Training SE')
			ax1[1].plot(lr, val_loss,color=color_pairs[i][1],label=labels[i]+' Val. SE')
			ax1[1].set_xlabel('Learning Rate')
			ax1[1].set_ylabel('SE')
			ax1[1].legend()
			ax1[1].set_xscale('log')
			ax1[1].grid()


	return ax



def extract_all_param_info():

    new_params = [
        'main_deflector_parameters_gamma','main_deflector_parameters_theta_E',
        'main_deflector_parameters_e1','main_deflector_parameters_e2',
        'source_parameters_R_sersic','source_parameters_n_sersic',
        'source_parameters_mag_app','lens_light_parameters_R_sersic',
        'lens_light_parameters_n_sersic','lens_light_parameters_mag_app',
        'point_source_parameters_mag_app','source_parameters_center_x',
        'source_parameters_center_y','main_deflector_parameters_center_x',
        'main_deflector_parameters_center_y','lens_light_parameters_e1',
        'lens_light_parameters_e2','source_parameters_e1',
        'source_parameters_e2','main_deflector_parameters_gamma1',
        'main_deflector_parameters_gamma2','point_source_parameters_mag_pert_0',
        'point_source_parameters_mag_pert_1','point_source_parameters_mag_pert_2',
        'point_source_parameters_mag_pert_3','main_deflector_parameters_caustic_area',
        'point_source_parameters_num_images','source_parameters_distance_to_caustic',
        'point_source_parameters_magnification_0','point_source_parameters_magnification_1',
        'point_source_parameters_magnification_2','point_source_parameters_magnification_3']

    """
    # VALIDATION SET

    y_test_val = np.load('../../STRIDES14results/april5M/val_y_test.npy')
    y_pred_val = np.load('../../STRIDES14results/april5M/val_y_pred.npy')
    std_pred_val = np.load('../../STRIDES14results/april5M/val_std_pred.npy')
    prec_pred_val = np.load('../../STRIDES14results/april5M/val_prec_pred.npy')

    tfr_test_path = '../../STRIDES14results/april2023/validate/data.tfrecord'
    tf_dataset_val = dataset_generation.generate_tf_dataset(tfr_test_path,new_params,
        10000,1,norm_images=True,kwargs_detector=None,
        input_norm_path=None,log_learning_params=None)
    
    # use iterator to grab truth values
    for batch in tf_dataset_val:
        param_truth_val = batch[1].numpy()

    # sort on gamma_lens to account for shuffling in tf dataset generation
    y_test_val_sort = np.argsort(y_test_val[:,3])
    param_truth_val_sort = np.argsort(param_truth_val[:,0])
    # apply sorting
    y_test_val = y_test_val[y_test_val_sort,:]
    y_pred_val = y_pred_val[y_test_val_sort,:]
    std_pred_val = std_pred_val[y_test_val_sort,:]
    prec_pred_val = prec_pred_val[y_test_val_sort,:,:]
    param_truth_val = param_truth_val[param_truth_val_sort,:]
    """


    # DOPPELGANGER SET

    doppel_names = ['ATLASJ2344-3056', 'DESJ0405-3308','DESJ0420-4037',
        'J1131-4419', 'J2145+6345',
        'J2205-3727','PSJ1606-2333', 'SDSSJ0248+1913', 
        'W2MJ1042+1641', 'WG0214-2105', 'WISEJ0259-1635']
    doppel_txt = np.asarray(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])

    y_test_doppel = np.load('../../STRIDES14results/april5M/doppel_y_test.npy')
    y_pred_doppel = np.load('../../STRIDES14results/april5M/doppel_y_pred.npy')
    std_pred_doppel = np.load('../../STRIDES14results/april5M/doppel_std_pred.npy')
    prec_pred_doppel = np.load('../../STRIDES14results/april5M/doppel_prec_pred.npy')

    # load in truth values from metadata.csv
    param_truth_doppel = []
    for i,d in enumerate(doppel_names):
        df = pd.read_csv('notebook_data/doppel_images/'+d+'/metadata.csv')
        param_truth_doppel.append(df[new_params].to_numpy()[0])
    param_truth_doppel = np.asarray(param_truth_doppel)

    # sort on gamma_lens to account for shuffling in tf dataset generation
    y_test_doppel_sort = np.argsort(y_test_doppel[:,3])
    param_truth_doppel_sort = np.argsort(param_truth_doppel[:,0])
    # apply sorting
    y_test_doppel = y_test_doppel[y_test_doppel_sort,:]
    y_pred_doppel = y_pred_doppel[y_test_doppel_sort,:]
    std_pred_doppel = std_pred_doppel[y_test_doppel_sort,:]
    prec_pred_doppel = prec_pred_doppel[y_test_doppel_sort,:,:]
    param_truth_doppel = param_truth_doppel[param_truth_doppel_sort,:]
    doppel_txt = doppel_txt[param_truth_doppel_sort]

    # QUADS ONLY NARROW SET

    y_test_narrow = np.load('../../STRIDES14results/narrow_images_quads_only/narrow_y_test.npy')
    y_pred_narrow = np.load('../../STRIDES14results/narrow_images_quads_only/narrow_y_pred.npy')
    std_pred_narrow = np.load('../../STRIDES14results/narrow_images_quads_only/narrow_std_pred.npy')
    prec_pred_narrow = np.load('../../STRIDES14results/narrow_images_quads_only/narrow_prec_pred.npy')

    # load in truth values from metadata.csv
    param_truth_narrow = []
    df = pd.read_csv('../../STRIDES14results/narrow_images_quads_only/metadata.csv')
    for n in range(0,5000):
        param_truth_narrow.append(df[new_params].to_numpy()[n])
    param_truth_narrow = np.asarray(param_truth_narrow)

    # check sorting assumption
    assert(np.abs(np.sum(param_truth_narrow[:,0] - y_test_narrow[:,3])) < 1e-5)


    # BOTH DOUBLES AND QUADS NARROW SET

    y_test_narrow_both = np.load('../../STRIDES14results/narrow_images_both/narrow_y_test.npy')
    y_pred_narrow_both = np.load('../../STRIDES14results/narrow_images_both/narrow_y_pred.npy')
    std_pred_narrow_both = np.load('../../STRIDES14results/narrow_images_both/narrow_std_pred.npy')
    prec_pred_narrow_both = np.load('../../STRIDES14results/narrow_images_both/narrow_prec_pred.npy')

    # load in truth values from metadata.csv
    param_truth_narrow_both = []
    df = pd.read_csv('../../STRIDES14results/narrow_images_both/metadata.csv')
    for n in range(0,5000):
        param_truth_narrow_both.append(df[new_params].to_numpy()[n])
    param_truth_narrow_both = np.asarray(param_truth_narrow_both)

    # check sorting assumption
    assert(np.abs(np.sum(param_truth_narrow_both[:,0] - y_test_narrow_both[:,3])) < 1e-5)

    return (y_test_doppel,y_pred_doppel,std_pred_doppel,prec_pred_doppel,param_truth_doppel,doppel_txt,
            y_test_narrow,y_pred_narrow,std_pred_narrow,prec_pred_narrow,param_truth_narrow,
            y_test_narrow_both,y_pred_narrow_both,std_pred_narrow_both,prec_pred_narrow_both,param_truth_narrow_both)


def calculate_info(param_truth,y_test,y_pred,std_pred):
    gamma_err = y_pred[:,3] - y_test[:,3] 
    mean_stds_away = np.mean(np.abs((y_pred[:,:6] - y_test[:,:6])/std_pred[:,:6])
                             ,axis=1)
    
    src_x = param_truth[:,11]
    src_y = param_truth[:,12]
    lens_x = param_truth[:,13]
    lens_y = param_truth[:,14]
    gamma = param_truth[:,0]
    mag_perts = param_truth[:,21:25]

    x_offset = (lens_x - src_x)

    y_offset = (lens_y - src_y)

    offset = np.sqrt( (src_x - lens_x)**2 + (src_y - lens_y)**2 )

    ellip = np.sqrt( param_truth[:,2]**2 + param_truth[:,3]**2 )

    shear = np.sqrt( param_truth[:,19]**2 + param_truth[:,20]**2)

    avg_percent_abs_mag_pert = np.mean(np.abs(1-mag_perts),axis=1)
    avg_percent_mag_pert = np.mean(1-mag_perts,axis=1)
    max_abs_mag_pert = np.max(np.abs(1-mag_perts),axis=1)
    # magnification 
    avg_mag = np.mean(np.abs(param_truth[:,28:32])/mag_perts[:,:],axis=1)
    # account for 1/2/3 image cases
    mask1 = np.where(np.isnan(mag_perts[:,3]))[0]
    if len(mask1) > 0:
        avg_percent_abs_mag_pert[mask1] = np.mean(np.abs(1-mag_perts[mask1,0:3]),axis=1)
        avg_percent_mag_pert[mask1] = np.mean(1-mag_perts[mask1,0:3],axis=1)
        max_abs_mag_pert[mask1] = np.max(np.abs(1-mag_perts[mask1,0:3]),axis=1)
        avg_mag[mask1] = np.mean(np.abs(param_truth[mask1,28:31])/mag_perts[mask1,0:3],axis=1)
    mask2 = np.where(np.isnan(mag_perts[:,2]))[0]
    if len(mask2) > 0:
        avg_percent_abs_mag_pert[mask2] = np.mean(np.abs(1-mag_perts[mask2,0:2]),axis=1)
        avg_percent_mag_pert[mask2] = np.mean(1-mag_perts[mask2,0:2],axis=1)
        max_abs_mag_pert[mask2] = np.max(np.abs(1-mag_perts[mask2,0:2]),axis=1)
        avg_mag[mask2] = np.mean(np.abs(param_truth[mask2,28:30])/mag_perts[mask2,0:2],axis=1)
    mask3 = np.where(np.isnan(mag_perts[:,1]))[0]
    if len(mask3) > 0:
        avg_percent_abs_mag_pert[mask3] = np.abs(1-mag_perts[mask3,0])
        avg_percent_mag_pert[mask3] = 1-mag_perts[mask3,0]
        max_abs_mag_pert[mask3] = np.abs(1-mag_perts[mask3,0])
        avg_mag[mask3] = np.abs(param_truth[mask3,28])/mag_perts[mask3,0]

    # sqrt(caustic_area) - 
    caustic_area = param_truth[:,25]
    caustic_vs_offset_length = np.sqrt(caustic_area) - offset

    return {
         'gamma_err':gamma_err,
         'mean_stds_away':mean_stds_away,
         'gamma':gamma,
         'x_offset':x_offset,
         'y_offset':y_offset,
         'offset':offset,
         'ellip':ellip,
         'shear':shear,
         'mag_perts':mag_perts,
         'avg_percent_abs_mag_pert':avg_percent_abs_mag_pert,
         'avg_percent_mag_pert':avg_percent_mag_pert,
         'max_abs_mag_pert':max_abs_mag_pert,
         'caustic_area':caustic_area,
         'caustic_vs_offset_length':caustic_vs_offset_length,
         'distance_to_caustic':param_truth[:,27],
         'avg_magnification':avg_mag
    } 



def clamp(val, minimum=0, maximum=255):
    # copied from: https://thadeusb.com/weblog/2010/10/10/python_scale_hex_color/
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return int(val)

def scale_hex_color(hexstr, scalefactor):
    """
    copied from: https://thadeusb.com/weblog/2010/10/10/python_scale_hex_color/
    Scales a hex string by ``scalefactor``. Returns scaled hex string.

    To darken the color, use a float value between 0 and 1.
    To brighten the color, use a float value greater than 1.

    >>> colorscale("#DF3C3C", .5)
    #6F1E1E
    >>> colorscale("#52D24F", 1.6)
    #83FF7E
    >>> colorscale("#4F75D2", 1)
    #4F75D2
    """

    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return "#%02x%02x%02x" % (r, g, b)


#####################
# corr. coeff. plots
#####################

def corr_coeff_plots(file_path,y_bounds,chosen_epochs,xlim=None):
    """ See how correlation coeffs progress as training goes on
    see corr_coeff_log.py to see how this is computed

    Args:
        file_path (string): path to log.csv computed using corr_coeff_log.py
        y_bounds ([(low,high)]): lower lim and upper lim tuple for each parameter
        chosen_epochs ([int]): puts a star where a chosen epoch is
        xlim ([low,high]): lower lim and upper lim for how many steps plotted,
            same for every parameter
    """

    # helper function to read csv columns
    def read_column(file_path,param_name):
        df = pd.read_csv(file_path)
        return df[param_name].to_numpy()

    fig,axs = plt.subplots(2,3,figsize=(18,12))
    plt.rc('font', **{'size':12})

    learning_params = ['main_deflector_parameters_theta_E','main_deflector_parameters_gamma1',
                    'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
                    'main_deflector_parameters_e1','main_deflector_parameters_e2']
                    #'main_deflector_parameters_center_x','main_deflector_parameters_center_y']

    learning_params_print = [r'$\theta_\mathrm{E}$',r'$\gamma_1$',r'$\gamma_2$',
                             r'$\gamma_\mathrm{lens}$',r'$e_1$',r'$e_2$']

    #y_bounds = [
    #    [0.93,1.0],[0.9,0.97],[0.9,0.97],[0.75,0.82],[0.9,0.97],[0.9,0.97]
    #]

    for i,l in enumerate(learning_params):
        corr_5e5 = read_column(file_path,l)
        steps_5e5 = (np.arange(len(corr_5e5))+1) * (5e5/512)
        idxs = np.arange(0,len(corr_5e5),1)
        axs[i//3,i%3].plot(steps_5e5[idxs],corr_5e5[idxs],label='5e5',color='blue')
        axs[i//3,i%3].legend()
        axs[i//3,i%3].set_title(learning_params_print[i])
        axs[i//3,i%3].set_xlabel('Steps')
        axs[i//3,i%3].set_xscale('log')
        if xlim is not None:
            axs[i//3,i%3].set_xlim(xlim)
        axs[i//3,i%3].set_ylim(y_bounds[i])\
        
        # display where I've chosen the weights
        for j,chosen_epoch in enumerate(chosen_epochs):
            chosen_step = chosen_epoch * (5e5/512)
            axs[i//3,i%3].scatter(chosen_step,corr_5e5[chosen_epoch-1],marker='*',
                color='magenta',zorder=200,label='epoch %d'%(chosen_epoch),s=100)
            
        axs[i//3,i%3].legend()


    plt.suptitle('Correlation Coefficients: july31_2023_lognorm')

# helper for combine_calib_plots
def construct_samps(y_pred,cov_pred):
    """
    Args:
        y_pred: (n_lenses,n_params)
        cov_pred: (n_lenses,n_params,n_params)
    """
    samps_array = np.empty(shape=(1000,y_pred.shape[0],y_pred.shape[1]))

    for p in range(0,len(y_pred)):
        samps = multivariate_normal(mean=y_pred[p],cov=cov_pred[p]).rvs(size=1000)
        samps_array[:,p,:] = samps

    return samps_array


def combine_calib_plots(y_pred_list,cov_pred_list,y_test,color_list,label_list,
    plot_title='Calibration Curve',save_path=None):
    """Combines paltas plot_calibration() for a list of y_pred,std_pred 
    on one figure

    Args:
        y_pred_list: (n_curves,n_lenses,n_params)
        cov_pred_list: (n_curves,n_lenses,n_params,n_params)
        y_test: ground truth for each lens, each param(n_lenses,n_params)
        color_list:
        label_list ([string]):
    """
     
    samps_init = construct_samps(y_pred_list[0],cov_pred_list[0])
    calib_figure = posterior_functions.plot_calibration(samps_init,y_test,
        show_plot=False,color_map=['black',color_list[0]],dpi=300)

    labels = ['SNPE m=0','SNPE m=1','SNPE m=2','SNPE m=4']
    legend_elements = ['Perfect Calibration','NPE']

    for j in range(1,len(y_pred_list)):
        samps = construct_samps(y_pred_list[j],cov_pred_list[j])

        if j == (len(y_pred_list)-1):
            posterior_functions.plot_calibration(samps,y_test,figure=calib_figure,
                color_map=['black',color_list[j]],legend=label_list,loc='upper left',
                title=plot_title,show_plot=False)
        else:
            posterior_functions.plot_calibration(samps,y_test,figure=calib_figure,
                color_map=['black',color_list[j]],show_plot=False)
             

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()