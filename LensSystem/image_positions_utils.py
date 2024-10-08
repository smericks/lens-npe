import os
import matplotlib as mpl
from LensSystem.paltas_model_image import PaltasModelImage
from LensSystem.forward_model_image import ForwardModelImage
import copy
from scipy.stats import norm as norm_dist
from scipy.stats import multivariate_normal
import paltas.Sampling.distributions as dist
from paltas.Sampling.sampler import Sampler
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from astropy.io import fits
from PIL import Image
from astropy.coordinates import SkyCoord
from visualization_utils import simple_norm
import shutil
from astropy.wcs import WCS
from corner.core import hist2d
from lenstronomy.SimulationAPI.data_api import DataAPI
import h5py

def construct_paltas_model(y_pred):
    """
    Args:
        y_pred ([float]): PEMD+Shear lens parameters
    """
     # create paltas object based on training config
    learned_model = PaltasModelImage('broad_training/training_config_broad.py')
    config_dict_copy = copy.deepcopy(learned_model.config_handler.config_dict)
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
    
    learned_model.config_handler.config_dict = config_dict_copy
    learned_model.config_handler.sampler = Sampler(config_dict_copy)

    _,_ = learned_model.config_handler.draw_image()

    return learned_model

def image_positions_arcsec(y_pred,data_correction=True):
    """
    Args:
        y_pred ([float]): PEMD+Shear lens parameters
    """
    learned_model = construct_paltas_model(y_pred)
    # returns [x_im,y_im]
    [x_im,y_im] = learned_model.image_positions()
    if data_correction:
            x_im = -(x_im-0.02)
            y_im = -(y_im+0.02)

    return x_im,y_im

# helper for the above function
def image_positions_grid(y_pred,data_correction=True):
    """
    Args:
        y_pred ([float]): PEMD+Shear lens parameters
        data_correction (bool): whether to apply pixel grid correction
    """
    learned_model = construct_paltas_model(y_pred)

    x_im,y_im = learned_model.image_positions_grid(data_correction=data_correction)

    return x_im, y_im

def grid_position_from_arcsec_position(x_im,y_im,data_correction=False):

    y_pred_static = [1.0,0.0,0.0,2.0,0.0,0.0,0.0,0.0,0.1,-0.1]
    paltas_object = construct_paltas_model(y_pred_static)

    sample = paltas_object.config_handler.get_current_sample()
    kwargs_detector = sample['detector_parameters']
    data_api = DataAPI(numpix=paltas_object.config_handler.numpix,**kwargs_detector)

    if data_correction:
            x_im = -(x_im-0.02)
            y_im = -(y_im+0.02)
    grid_x, grid_y = data_api.data_class.map_coord2pix(x_im,y_im)
    return grid_x,grid_y

def list_image_positions(fm_file,y_pred):
    """
        fm_file (string): 
        y_pred ([float]): lens model predicted means
    """

    # read off image positions from the .results file
    fm = ForwardModelImage(fm_file,make_image=True)
    kwargs_params = fm.kwargs_result
    image_positions_ps = fm.point_source_model.image_position(
            kwargs_ps=kwargs_params['kwargs_ps'],
            kwargs_lens=kwargs_params['kwargs_lens'])
    x_im_truth = image_positions_ps[0][0]
    y_im_truth = image_positions_ps[1][0]

    if fm_file=='../doppelgangers/'+'J2145+6345'+'_results.txt':
        x_im_truth += 0.16
        y_im_truth -= 0.32

    return x_im_truth,y_im_truth

def matrix_plot_im_positions(y_pred,image_folder,df,indices,dim,save_name,PlotCenter=False,
                show_one_arcsec=True,cov_pred=None,fm_files_for_astrometry=None):
    """
    Args: 
        y_pred ([float]): lens model predicted means
        image_folder (string): Name/path of folder storing .fits files
        df (pandas dataframe): Dataframe containing all info from lens catalog 
            .csv file
        indices (numpy array): List of indices of which systems to plot
        dim (int,int): Tuple of (number rows, number columns), defines shape of 
            matrix plotted
        save_name (string): Filename to save final image to
        PlotCenter (boolean): If True, plots small red star in center of image
        show_one_arcsec (boolean): If True, plots 1 arcsec bar in top left corner
        cov_pred ([]): lens model predicted covariance matrices
        fm_files_for_astrometry ([string]): if not None, loads in image positions
            from forward modeling and details the deviation
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
            plt.figure(dpi=300)
            # fill empty spot (index = -1)
            if file_idx == -1:
                plt.matshow(np.ones((100,100)),cmap='Greys_r')
                plt.axis('off')
                plt.savefig(intm_name,bbox_inches='tight',pad_inches=0,dpi=300)
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
            plt.matshow(cropped_data,cmap='viridis',norm=norm)
            if PlotCenter:
                plt.scatter(offset,offset,edgecolors='red',marker='*',
                    facecolors='none',s=100)
                
            if y_pred[file_counter][0] > 0:
                # ADD IMAGE POSITIONS
                #if std_pred is None:
                x_im,y_im = image_positions_grid(y_pred[file_counter])
                plt.scatter(x_im,y_im,edgecolors='red',marker='*',facecolors='none',
                    s=200,linewidths=3,zorder=100)
                
                # TODO: if astrometry requested, do it!
                if fm_files_for_astrometry is not None:
                    fm_file = fm_files_for_astrometry[file_counter]
                    x_im_truth,y_im_truth = list_image_positions(
                        fm_file,y_pred[file_counter])

                    grid_x,grid_y = grid_position_from_arcsec_position(x_im_truth,y_im_truth,data_correction=True)


                    # matching images truth to images pred
                    x_im_pred = []
                    y_im_pred = []
                    for k in range(0,len(grid_x)):
                        try:
                            idx = np.where((np.abs(grid_x[k] - x_im) < 3.5) & 
                                        (np.abs(grid_y[k] - y_im) < 3.5))[0][0]
                            
                            x_im_pred.append(x_im[idx])
                            y_im_pred.append(y_im[idx])
                        except:
                            # TODO: edge case: where there's no matching imag
                            x_im_pred.append(np.nan)
                            y_im_pred.append(np.nan)


                    # let's just do this here
                    pix_to_arcsec = 0.04
                    x_diff = (x_im_pred-grid_x)*0.04
                    y_diff = (y_im_pred-grid_y)*0.04
                    diff = np.sqrt(x_diff**2 + y_diff**2)

                    # what if we just plot grid_x,grid_y to debug
                    my_avg = []
                    for im in range(0,4):
                        #plt.text(grid_x[im]-3,grid_y[im]-3,'%.2f'%(diff[im]),color='red',fontsize=20)
                        if not np.isnan(diff[im]):
                            my_avg.append(diff[im]*1000)

                    plt.text(28,75,'$\delta\\theta$ = %d mas'%(np.mean(my_avg)),
                        color='white',fontsize=20)
                        #plt.text(grid_x[im],grid_y[im],'',color='red',fontsize=10)

                    # write distance btwn. image and truth
                    #to_avg = []
                    #for ti in range(0,len(distance_to_truth)):
                    #    if not np.isnan(distance_to_truth[ti]):
                    #        to_avg.append(distance_to_truth[ti])
                            #plt.text(grid_x[ti]+3,grid_y[ti]+3,
                            #    '%.2f'%(distance_to_truth[ti]),color='red',
                            #    fontsize=15)

                    #plt.text(50,70,'%.2f "'%(np.mean(to_avg)),color='red',fontsize=24)
                        
                if cov_pred is not None:
                    y_pred_samples = multivariate_normal(mean=y_pred[file_counter],
                        cov=cov_pred[file_counter]).rvs(size=10)
                    x_im_list = []
                    y_im_list = []
                    for yp in y_pred_samples:
                        x_im,y_im = image_positions_grid(yp)
                        x_im_list.extend(x_im)
                        y_im_list.extend(y_im)
                        plt.scatter(x_im,y_im,edgecolors='coral',marker='.',facecolors='none',
                            s=100,linewidths=3)
                    #axs = plt.gca()
                    #print(np.min(np.asarray(y_im_list)))
                    #print(np.max(np.asarray(y_im_list)))
                    #hist2d(np.asarray(x_im_list),np.asarray(y_im_list),ax=axs,
                    #       color='orange',levels=[0.68],
                    #       plot_datapoints=False,plot_density=False,
                    #       range=([0,80],[0,80]),force_range=True)
        

            # annotate system name
            plt.annotate(system_names[file_idx],(2*offset - offset/8,offset/6),color='white',
                fontsize=20,horizontalalignment='right')
            # show size of 1 arcsec
            if show_one_arcsec:
                plt.plot([offset/6,offset/6],[offset/8,
                    offset/8 + (1/arcseconds_per_pixel)],color='white')
                if file_counter == 0:
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


# TODO: GET RIDE OF CODE DUPLICATION
def matrix_plot_im_positions_h5(y_pred,image_h5_path,indices,dim,save_name,
                show_one_arcsec=True,cov_pred=None,fm_files_for_astrometry=None):
    """
    Args: 
        y_pred ([float]): lens model predicted means
        image_h5_path (string): Name/path of .h5 file storing images
        df (pandas dataframe): Dataframe containing all info from lens catalog 
            .csv file
        indices (numpy array): List of indices of which systems to plot
        dim (int,int): Tuple of (number rows, number columns), defines shape of 
            matrix plotted
        save_name (string): Filename to save final image to
        show_one_arcsec (boolean): If True, plots 1 arcsec bar in top left corner
        cov_pred ([]): lens model predicted standard deviations
        fm_files_for_astrometry ([string]): if not None, loads in image positions
            from forward modeling and details the deviation
    """

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

    # open .h5 file!
    # load in .h5 file
    f = h5py.File(image_h5_path, "r")
    f_data = f['data'][()]

    # iterate through matrix & fill each image
    for i in range(0,dim[0]):
        row = []
        for j in range(0,dim[1]):
            
            file_idx = int(indices[file_counter])

            # fill empty spot (index = -1)
            if file_idx == -1:
                plt.matshow(np.ones((100,100)),cmap='Greys_r')
                plt.axis('off')
                plt.savefig(intm_name,bbox_inches='tight',pad_inches=0,dpi=300)
                img_data = np.asarray(Image.open(intm_name))
                row.append(img_data)
                continue

            cropped_data = f_data[file_idx]

            # normalize data using log and min cutoff 
            norm = simple_norm(cropped_data,stretch='log',min_cut=1e-6)
        
            # create individual image using plt library
            plt.matshow(cropped_data,cmap='viridis',norm=norm)
                
            if y_pred[file_idx][0] > 0:
                # ADD IMAGE POSITIONS
                #if std_pred is None:
                x_im,y_im = image_positions_grid(y_pred[file_idx],data_correction=False)
                plt.scatter(x_im,y_im,edgecolors='red',marker='*',facecolors='none',
                    s=200,linewidths=3,zorder=100)
                        
                if cov_pred is not None:
                    y_pred_samples = multivariate_normal(mean=y_pred[file_idx],
                        cov=cov_pred[file_idx]).rvs(size=10)
                    x_im_list = []
                    y_im_list = []
                    for yp in y_pred_samples:
                        x_im,y_im = image_positions_grid(yp,data_correction=False)
                        x_im_list.extend(x_im)
                        y_im_list.extend(y_im)
                        plt.scatter(x_im,y_im,edgecolors='coral',marker='.',facecolors='none',
                            s=100,linewidths=3)
    
            plt.axis('off')

            # save intermediate file, then read back in as array, and save to row
            intm_name = ('intermediate_temp/'+str(file_counter)+'.png')
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

    # close the .h5 file
    f.close()

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
    plt.figure(figsize=(2*dim[1],2*dim[0]))
    plt.imshow(final_image)
    plt.axis('off')
    plt.savefig(save_name,bbox_inches='tight')
    plt.show()