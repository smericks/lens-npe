# given a set of model weights, produce model predictions 
# WARNING: heavily hard-coded for my specific use case
from paltas.Analysis import dataset_generation, loss_functions, conv_models
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/deep-lens-modeling/')
from LensSystem.hst_image import HSTImage
import h5py


class NetworkPredictions():
    """
    Given a trained network, generate & store all predictions for different
    test sets
    The idea is you would have one of these objects for each model weights file
    """
    
    def __init__(self,path_to_model_weights,path_to_model_norms=None,
                 learning_params=None,loss_type='full',model_type='xresnet34',
                 norm_type='norm'):
        """
        Args:
            path_to_model_weights (string)
            path_to_model_norms (string)
            loss_type: 'full' and 'diag' supported
            model_type: 'xresnet34' and 'xresnet101' supported
            norm_type: 'norm' and 'lognorm' supported
                - If 'norm', normalize test set images to range (0,1)
                - If 'lognorm', test set imags are log-normalized and 
                  then rescaled to range (0,1)
        """
        img_size = (80,80,1)
        # do we want this the same every time? good for reproducing the same results...
        random_seed = 2
        batch_size = 512
        flip_pairs = None

        if norm_type in {'norm','lognorm'}:
            self.norm_type = norm_type
            self.norm_images = (norm_type == 'norm')
            self.lognorm_images = (norm_type == 'lognorm')
        else:
            raise ValueError('norm_type not supported in NetworkPredictions initialization')
        
        if learning_params is None:

            self.learning_params = ['main_deflector_parameters_theta_E','main_deflector_parameters_gamma1',
                            'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
                            'main_deflector_parameters_e1','main_deflector_parameters_e2',
                            'main_deflector_parameters_center_x','main_deflector_parameters_center_y']
                            #'source_parameters_center_x','source_parameters_center_y',
                            #'source_parameters_R_sersic']
        else:
            self.learning_params = learning_params

        log_learning_params = []
        num_params = len(learning_params+log_learning_params)
        self.loss_type = loss_type
        self.model_type = model_type

        tf.random.set_seed(random_seed)

        if self.loss_type == 'full':
            num_outputs = num_params + int(num_params*(num_params+1)/2)
            self.loss_func = loss_functions.FullCovarianceLoss(num_params)

        elif self.loss_type == 'diag':
            num_outputs = 2*num_params
            self.loss_func = loss_functions.DiagonalCovarianceLoss(num_params)

        else:
            raise ValueError('loss_type not supported in NetworkPredictions initialization')

        if model_type == 'xresnet101':
            self.model = conv_models.build_xresnet101(img_size,num_outputs)
        if model_type == 'xresnet34':
            self.model = conv_models.build_xresnet34(img_size,num_outputs)
        else:
            raise ValueError('model_type not supported in NetworkPredictions initialization')
        
        self.model.load_weights(path_to_model_weights)
        self.norm_path = path_to_model_norms
        #model.load_weights(model_weights,by_name=True,skip_mismatch=True)


    def _process_image_batch(self,images,samples=False,norm_path=None):
        """Generate network predictions given some batch of images

        Args:
            images
            samples (Bool)
            norm_path (string): if None and samples=True, throw error

        Returns:
            if samples is True:
                y_pred,std_pred,cov_mat,un_samples
            else:
                y_pred,std_pred,cov_mat

        """
        # use unrotated output for covariance matrix
        output = self.model.predict(images)

        if self.loss_type == 'full':
            y_pred, precision_matrix, _ = self.loss_func.convert_output(output)
        else:
            y_pred, log_var_pred = self.loss_func.convert_output(output)

        # compute std. dev.
        if self.loss_type == 'full':
            cov_mat = np.linalg.inv(precision_matrix.numpy())
            std_pred = np.zeros((cov_mat.shape[0],cov_mat.shape[1]))
            for i in range(len(std_pred)):
                std_pred[i] = np.sqrt(np.diag(cov_mat[i]))
                
        else:
            std_pred = np.exp(log_var_pred/2)
            cov_mat = np.empty((len(std_pred),len(std_pred[0]),len(std_pred[0])))
            for i in range(len(std_pred)):
                cov_mat[i] = np.diag(std_pred[i]**2)


        if samples:
            if norm_path is None:
                raise ValueError('need to define norm_path when generating samples in _process_image_batch()')
            un_samples = self.loss_func.draw_samples(output,n_samps=10000)
            # trying to unnormalize samples correctly 
            # unnormalize_outputs() expects shape (batchsize,num_params)
            for j in range(0,un_samples.shape[1]):
                dataset_generation.unnormalize_outputs(norm_path,
                    self.learning_params+[],un_samples[:,j,:])
            
            return y_pred,std_pred,cov_mat,un_samples

        return y_pred,std_pred,cov_mat


    ##############################
    # generate network predictions
    ##############################
    def gen_network_predictions(self,test_folder,samples=False,shuffle=False):
        """
        Generate neural network predictions given a paltas generated folder of images

        Args:
            test_folder (string): Path to folder of paltas generated images, 
                containig a data.tfrecord file
            samples (bool, default=False): If True, samples from the NPE are returned
            shuffle (bool, default=False): If True, the order of the test set is shuffled
                when generating predictions
            norm_images (bool, default=True): If True, normalize test set images
            log_norm_images (bool, default=False): If True, test set images are
                log-normalized and rescaled to range (0,1)

        Returns:
            If samples=True:
                y_test, y_pred, std_pred, prec_pred, predict_samps
            Else:
                y_test, y_pred, std_pred, prec_pred
        """

        npy_folder_test = test_folder
        tfr_test_path = os.path.join(npy_folder_test,'data.tfrecord')

        tf_dataset_test = dataset_generation.generate_tf_dataset(tfr_test_path,
            self.learning_params,3,1,norm_images=self.norm_images,
            log_norm_images=self.lognorm_images,kwargs_detector=None,
            input_norm_path=self.norm_path,log_learning_params=[],shuffle=shuffle)

        #tf_dataset_test = dataset_generation.generate_rotations_dataset(tfr_test_path,
        #    self.learning_params,3,1,norm_images=self.norm_images,
        #    log_norm_images=self.lognorm_images,kwargs_detector=None,
        #    input_norm_path=self.norm_path,log_learning_params=[],shuffle=shuffle)


        y_test_list = []
        y_pred_list = []
        std_pred_list = []
        cov_pred_list = []
        predict_samps_list = []

        for batch in tf_dataset_test:

            #images = batch[0]
            #y_test = batch[1]
            images = batch[0].numpy()
            y_test = batch[1].numpy()
            
            if samples:
                y_pred,std_pred,cov_mat,un_samples = self._process_image_batch(images,
                    True,self.norm_path)
            else:
                y_pred,std_pred,cov_mat = self._process_image_batch(images)

            y_test_list.append(y_test)
            y_pred_list.append(y_pred)
            std_pred_list.append(std_pred)
            cov_pred_list.append(cov_mat)
            if samples:
                predict_samps_list.append(un_samples)

        y_test = np.concatenate(y_test_list)
        y_pred = np.concatenate(y_pred_list)
        std_pred = np.concatenate(std_pred_list)
        cov_pred = np.concatenate(cov_pred_list)
        if samples:
            predict_samps = np.concatenate(predict_samps_list,axis=1)


        dataset_generation.unnormalize_outputs(self.norm_path,self.learning_params+[],
            y_pred,standard_dev=std_pred,cov_mat=cov_pred)
        dataset_generation.unnormalize_outputs(self.norm_path,self.learning_params+[],
            y_test)
                                            
        prec_pred = np.linalg.inv(cov_pred)
        # fixing np.linalg.inv numerical problems by taking the average - Padma V. 5/30/24
        for i in range(prec_pred.shape[0]):
            prec_pred[i] = prec_pred[i, :, :]/2 + prec_pred[i, :, :].T/2

        if samples:
            return y_test, y_pred, std_pred, prec_pred, predict_samps
        
        return y_test, y_pred, std_pred, prec_pred
    
    def return_y_test_output(self,test_folder):

        npy_folder_test = test_folder
        tfr_test_path = os.path.join(npy_folder_test,'data.tfrecord')
        tf_dataset_test = dataset_generation.generate_tf_dataset(tfr_test_path,
            self.learning_params,5000,1,norm_images=self.norm_images,
            log_norm_images=self.lognorm_images,kwargs_detector=None,
            input_norm_path=self.norm_path,log_learning_params=[])

        # only one thing in tf_dataset_test since batch size = 5,000
        for t in tf_dataset_test:
            batch = t
        images = batch[0].numpy()
        y_test_d = batch[1].numpy()

        output = self.model.predict(images)

        return y_test_d, output




    def gen_doppel_predictions(self,test_folder,doppel_names,samples=False):
        
        """
        Generate neural network predictions given a folder of individual paltas folders
        (specifically useful for doppelganger format)

        Args:
            test_folder (string): Path to folder of folders, where each folder is for an 
                individual paltas generated images, containig a data.tfrecord file
            doppel_names (list[string]): Names of the sub-folders
            norm_path (string): Path to .csv containing normalization of parameters
                applied during training of network
            samples (bool, default=False): If True, samples from the NPE are returned

        Returns:
            If samples=True:
                y_test, y_pred, std_pred, prec_pred, predict_samps
            Else:
                y_test, y_pred, std_pred, prec_pred
        """
        

        y_test_doppel = []
        y_pred_doppel = []
        std_pred_doppel = []
        cov_pred_doppel = []
        predict_samps_doppel = []


        for i,doppel in enumerate(doppel_names):
            npy_folder_test = test_folder+doppel+'/'
            tfr_test_path = os.path.join(npy_folder_test,'data.tfrecord')

            tf_dataset_test = dataset_generation.generate_tf_dataset(tfr_test_path,
                self.learning_params,3,1,norm_images=self.norm_images,
                log_norm_images=self.lognorm_images,kwargs_detector=None,
                input_norm_path=self.norm_path,log_learning_params=[])

            # only one thing in tf_dataset_test since individual doppelgangers
            for t in tf_dataset_test:
                batch = t
            images = batch[0].numpy()
            y_test_d = batch[1].numpy()

            # use unrotated output for covariance matrix
            if samples:
                y_pred,std_pred,cov_mat,un_samples = self._process_image_batch(images,
                        True,self.norm_path)
            else:
                y_pred,std_pred,cov_mat = self._process_image_batch(images)

            y_test_doppel.append(y_test_d)
            y_pred_doppel.append(y_pred)
            std_pred_doppel.append(std_pred)
            cov_pred_doppel.append(cov_mat)
            if samples:
                predict_samps_doppel.append(un_samples)


        y_test_doppel = np.concatenate(y_test_doppel)
        y_pred_doppel = np.concatenate(y_pred_doppel)
        std_pred_doppel = np.concatenate(std_pred_doppel)
        cov_pred_doppel = np.concatenate(cov_pred_doppel)
        if samples:
            predict_samps_doppel = np.concatenate(predict_samps_doppel,axis=1)

        dataset_generation.unnormalize_outputs(self.norm_path,self.learning_params+[],
                                            y_pred_doppel,standard_dev=std_pred_doppel,
                                            cov_mat=cov_pred_doppel)
        dataset_generation.unnormalize_outputs(self.norm_path,self.learning_params+[],
                                            y_test_doppel)

        # get back to prec matrix
        # np.linalg.inv expects matrix dimensions as last 2 dimensions (11,8,8)
        prec_pred_doppel = np.linalg.inv(cov_pred_doppel)
        # fixing np.linalg.inv numerical problems by taking the average - Padma V. 5/30/24
        for i in range(prec_pred_doppel.shape[0]):
            prec_pred_doppel[i] = prec_pred_doppel[i, :, :]/2 + prec_pred_doppel[i, :, :].T/2

        if samples:
            return y_test_doppel, y_pred_doppel, std_pred_doppel, prec_pred_doppel, predict_samps_doppel
        
        return y_test_doppel, y_pred_doppel, std_pred_doppel, prec_pred_doppel 
    

    def gen_data_predictions(self,fits_file_list,lens_names_list,samples=False):
        
        """
        Generate neural network predictions given .fits files

        Args:
            fits_file_list ([string]): list of sci.fits files of HST observations
            lens_names_list ([string]): list of names of lenses to retrieve from
                csv catalog
            samples (bool, default=False): If True, samples from the NPE are returned

        Returns:
            If samples=True:
                y_pred, std_pred, prec_pred, predict_samps
            Else:
                y_pred, std_pred, prec_pred
        """

        y_pred_data = []
        std_pred_data = []
        cov_pred_data = []
        predict_samps_data = []


        for i,file_name in enumerate(fits_file_list):

            catalog_path = ('https://docs.google.com/spreadsheets/d/'+
                '1jOC60bWMxpp65iJZbANc_6SxouyXwqsESF4ocLAj27E/export?gid=0&format=csv')
            catalog_df = pd.read_csv(catalog_path)
            lens_catalog_row = catalog_df[catalog_df['name'] == 
                lens_names_list[i]].iloc[0]

            hst_im = HSTImage(file_name,lens_catalog_row)
            image = hst_im.image_data

            # normalize image if needed
            if self.norm_images:
                image = dataset_generation.norm_image(image)
            elif self.lognorm_images:
                image = dataset_generation.log_norm_image(image)

            # use unrotated output for covariance matrix
            # needs to be shape (1,80,80,1), can be numpy
            reshaped_tensor = tf.reshape(image,[1,80,80,1])

            # use unrotated output for covariance matrix
            if samples:
                y_pred,std_pred,cov_mat,un_samples = self._process_image_batch(reshaped_tensor,
                        True,self.norm_path)
            else:
                y_pred,std_pred,cov_mat = self._process_image_batch(reshaped_tensor)

            y_pred_data.append(y_pred)
            std_pred_data.append(std_pred)
            cov_pred_data.append(cov_mat)
            if samples:
                predict_samps_data.append(un_samples)

        y_pred_data = np.concatenate(y_pred_data)
        std_pred_data = np.concatenate(std_pred_data)
        cov_pred_data = np.concatenate(cov_pred_data)
        if samples:
            predict_samps_data = np.concatenate(predict_samps_data,axis=1)

        dataset_generation.unnormalize_outputs(self.norm_path,self.learning_params+[],
                                            y_pred_data,standard_dev=std_pred_data,
                                            cov_mat=cov_pred_data)

        # get back to prec matrix
        # np.linalg.inv expects matrix dimensions as last 2 dimensions (11,8,8)
        prec_pred_data = np.linalg.inv(cov_pred_data)
        # fixing np.linalg.inv numerical problems by taking the average - Padma V. 5/30/24
        for i in range(prec_pred_data.shape[0]):
            prec_pred_data[i] = prec_pred_data[i, :, :]/2 + prec_pred_data[i, :, :].T/2

        if samples:
            return y_pred_data, std_pred_data, prec_pred_data, predict_samps_data
        
        return y_pred_data, std_pred_data, prec_pred_data
    
    def loss_single_image(self,y_test,image):
        """ Takes in a single image/y_test pair and returns the loss value
            (equivalent to the negative log pdf when 'full' or 'diagonal' are used)

        Args:
            y_test (np array with shape (1, 11))
            image (np array with shape (1, 80, 80, 1))

        Returns:
            loss value (float)
        
        """
        output = self.model.predict(image)

        loss = self.loss_func.loss(y_test,output)
        
        # do we need to index into loss / is it returned as a list?
        return loss.numpy()[0]




    def write_preds_to_h5(self,write_path,y_pred,std_pred,prec_pred,y_test=None,predict_samps=None):
        """
        Args:
            write_path (string)
        """

        h5f = h5py.File(write_path, 'w')
        h5f.create_dataset('y_pred', data=y_pred)
        h5f.create_dataset('std_pred', data=std_pred)
        h5f.create_dataset('prec_pred', data=prec_pred)
        if y_test is not None:
            h5f.create_dataset('y_test', data=y_test)
        if predict_samps is not None:
            h5f.create_dataset('predict_samps', data=predict_samps)
        h5f.close()


    def generate_all_predictions(self,write_folder,path_to_val_set=None,
        path_to_narrow_set=None,path_to_doppelganger_set=None,doppel_names=None,
        fits_file_list=None,lens_names_list=None,path_to_training_set=None):
        """
        Args: 
            write_folder (string): folder where predictios are stored in .h5 
                files
            path_to_val_set (string): folder where validation set images are 
                stored
            path_to_narrow_set (string): folder where narrow test set images are 
                stored
            path_to_doppelganger_set (string): folder where doppelganger test 
                set images are stored (in separate folders)
            doppel_names ([string]): list of folder names of doppelgangers
            fits_file_list ([string]): list of .fits file for each data image
            lens_names_list ([string]): list of names corresponding to each data 
                image
            path_to_training_set (string): folder where training set images are 
                stored

        """

        # validation set
        if path_to_val_set is not None:
            (y_test, y_pred, std_pred, 
                prec_pred, predict_samps) = self.gen_network_predictions(path_to_val_set,samples=True)
            h5_path = write_folder + 'validation_predictions.h5'
            self.write_preds_to_h5(h5_path,y_pred,std_pred,prec_pred,y_test,predict_samps)
        # training set
        if path_to_training_set is not None:
            (y_test, y_pred, std_pred, 
                prec_pred, predict_samps) = self.gen_network_predictions(path_to_training_set,samples=True)
            h5_path = write_folder + 'training_predictions.h5'
            self.write_preds_to_h5(h5_path,y_pred,std_pred,prec_pred,y_test,predict_samps)
        # narrow test set
        if path_to_narrow_set is not None:
            (y_test, y_pred, std_pred, 
                prec_pred, predict_samps) = self.gen_network_predictions(path_to_narrow_set,samples=True)
            h5_path = write_folder + 'narrow_predictions.h5'
            self.write_preds_to_h5(h5_path,y_pred,std_pred,prec_pred,y_test,predict_samps)
        # doppelganger test set
        if path_to_doppelganger_set is not None:
            (y_test, y_pred, std_pred, 
                prec_pred, predict_samps) = self.gen_doppel_predictions(path_to_doppelganger_set,doppel_names,samples=True)
            h5_path = write_folder + 'doppelganger_predictions.h5'
            self.write_preds_to_h5(h5_path,y_pred,std_pred,prec_pred,y_test,predict_samps)
        # HST data test set
        if fits_file_list is not None:
            (y_pred, std_pred, prec_pred, 
                predict_samps) = self.gen_data_predictions(fits_file_list,lens_names_list,samples=True)
            h5_path = write_folder + 'HSTdata_predictions.h5'
            self.write_preds_to_h5(h5_path,y_pred,std_pred,prec_pred,y_test=None,predict_samps=predict_samps)


def generate_data_sequential_predictions(weights_path_list,fits_file_list,lens_name_list,
    norms_path,loss_type='full'):
    """
    Args:
        weights_path_list ([string]): list of network weights 
            (one for each doppel)
        fits_file_list ([string]): list of sci.fits files of HST observations
        lens_names_list ([string]): list of names of lenses to retrieve from
            csv catalog
        norms_path (string): single norms.csv path, same for all
        loss_type (string): 'full' or 'diag' supported

    Returns: 
        y_pred, std_pred, prec_pred
    """
    
    learning_params = ['main_deflector_parameters_theta_E','main_deflector_parameters_gamma1',
                    'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
                    'main_deflector_parameters_e1','main_deflector_parameters_e2',
                    'main_deflector_parameters_center_x','main_deflector_parameters_center_y',
                    'source_parameters_center_x','source_parameters_center_y']

    y_pred_d = []
    std_pred_d = []
    prec_pred_d = []

    for i in range(0,len(weights_path_list)):
        predictor = NetworkPredictions(weights_path_list[i],
            path_to_model_norms=norms_path,
            learning_params=learning_params,loss_type=loss_type,
            model_type='xresnet34',norm_type='lognorm')
        
        print(lens_name_list[i])
        y_pred, std_pred, prec_pred = predictor.gen_data_predictions(
            fits_file_list=[fits_file_list[i]],lens_names_list=[lens_name_list[i]])

        y_pred_d.append(y_pred)
        std_pred_d.append(std_pred)
        prec_pred_d.append(prec_pred)

    y_pred_d = np.concatenate(y_pred_d)
    std_pred_d = np.concatenate(std_pred_d)
    prec_pred_d = np.concatenate(prec_pred_d)

    return y_pred_d, std_pred_d, prec_pred_d



def generate_doppel_sequential_predictions(weights_path_list,doppel_image_folder,
        doppel_names,norms_path,loss_type='full',image_type='h5'):
    """
    Args:
        weights_path_list: list of network weights (one for each doppel)
        narrow_image_folder (string): path to where test set images are stored
        ...
        norms_path (string): single norms.csv path, same for all
        loss_type (string): 'full' or 'diag' supported
    """

    learning_params = ['main_deflector_parameters_theta_E','main_deflector_parameters_gamma1',
                   'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
                   'main_deflector_parameters_e1','main_deflector_parameters_e2',
                   'main_deflector_parameters_center_x','main_deflector_parameters_center_y',
                   'source_parameters_center_x','source_parameters_center_y']

    y_pred_doppel = []
    std_pred_doppel = []
    cov_pred_doppel = []


    for i in range(0,len(doppel_names)):
        predictor = NetworkPredictions(weights_path_list[i],
            path_to_model_norms=norms_path,
            learning_params=learning_params,loss_type=loss_type,
            model_type='xresnet34',norm_type='lognorm')

        # load in the image
        image_path = doppel_image_folder+doppel_names[i]+'/image_0000000.npy'
        image = np.load(image_path)

        # NORMALIZE THE IMAGES!!!!! (to match training convention)
        image = dataset_generation.log_norm_image(image)
        
        y_pred_d,std_pred_d,cov_mat_d = predictor._process_image_batch(np.asarray([image]))

        y_pred_doppel.append(y_pred_d)
        std_pred_doppel.append(std_pred_d)
        cov_pred_doppel.append(cov_mat_d)
        

    y_pred_doppel = np.concatenate(y_pred_doppel)
    std_pred_doppel = np.concatenate(std_pred_doppel)
    cov_pred_doppel = np.concatenate(cov_pred_doppel)

    dataset_generation.unnormalize_outputs(norms_path,learning_params+[],
                                            y_pred_doppel,standard_dev=std_pred_doppel,
                                            cov_mat=cov_pred_doppel)

    # get back to prec matrix
    # np.linalg.inv expects matrix dimensions as last 2 dimensions (11,8,8)
    prec_pred_doppel = np.linalg.inv(cov_pred_doppel)

    return y_pred_doppel, std_pred_doppel, prec_pred_doppel


def generate_narrow_sequential_predictions(weights_path_list,narrow_image_folder,image_indices,
                                           norms_path,loss_type='diag',image_type='h5'):
    """
    Args:
        weights_path_list: list of network weights (one for each doppel)
        narrow_image_folder (string): path to where test set images are stored
        image_indices ([int]): which images in the test set to evaluate on
        ...
        norms_path (string): single norms.csv path, same for all
        loss_type (string): 'full' or 'diag' supported
    """

    learning_params = ['main_deflector_parameters_theta_E','main_deflector_parameters_gamma1',
                   'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
                   'main_deflector_parameters_e1','main_deflector_parameters_e2',
                   'main_deflector_parameters_center_x','main_deflector_parameters_center_y',
                   'source_parameters_center_x','source_parameters_center_y']

    y_pred_doppel = []
    std_pred_doppel = []
    cov_pred_doppel = []

    #extract array of images from .h5 file
    if image_type == 'h5':
        h5_file = narrow_image_folder+'image_data.h5'
        f = h5py.File(h5_file, "r")
        image_array = f['data'][()]

    for i in range(0,len(image_indices)):
        predictor = NetworkPredictions(weights_path_list[i],
            path_to_model_norms=norms_path,
            learning_params=learning_params,loss_type=loss_type,
            model_type='xresnet34',norm_type='lognorm')
        

        if image_type == 'h5':
            image = image_array[image_indices[i]]
        
        else:
            # load in the image
            j = image_indices[i]
            image_path = narrow_image_folder+'image_%07d.npy'%(j)
            image = np.load(image_path)

        # NORMALIZE THE IMAGES!!!!! (to match training convention)
        image = dataset_generation.log_norm_image(image)
        
        y_pred_d,std_pred_d,cov_mat_d = predictor._process_image_batch(np.asarray([image]))

        y_pred_doppel.append(y_pred_d)
        std_pred_doppel.append(std_pred_d)
        cov_pred_doppel.append(cov_mat_d)
        
    if image_type == 'h5':
        f.close()

    y_pred_doppel = np.concatenate(y_pred_doppel)
    std_pred_doppel = np.concatenate(std_pred_doppel)
    cov_pred_doppel = np.concatenate(cov_pred_doppel)

    dataset_generation.unnormalize_outputs(norms_path,learning_params+[],
                                            y_pred_doppel,standard_dev=std_pred_doppel,
                                            cov_mat=cov_pred_doppel)

    # get back to prec matrix
    # np.linalg.inv expects matrix dimensions as last 2 dimensions (11,8,8)
    prec_pred_doppel = np.linalg.inv(cov_pred_doppel)

    return y_pred_doppel, std_pred_doppel, prec_pred_doppel

def generate_sequential_predictions(weights_path_list,doppel_image_folder,
        doppel_name_list,norms_path,loss_type='full'):
    """
    Args:
        weights_path_list: list of network weights (one for each doppel)
        doppel_image_folder (string): path to where doppel folders live
        doppel_name_list [string]: list of names of folders for doppel images
        norms_path: single norms.csv path, same for all doppels
        loss_type (string): 'full' or 'diag'

    Returns:
        y_test_doppel, y_pred_doppel, std_pred_doppel, prec_pred_doppel

    Note:
        hardcoded to loss_type='full',model_type='xresnet34',norm_type='lognorm'
    """

    learning_params = ['main_deflector_parameters_theta_E','main_deflector_parameters_gamma1',
                   'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
                   'main_deflector_parameters_e1','main_deflector_parameters_e2',
                   'main_deflector_parameters_center_x','main_deflector_parameters_center_y',
                   'source_parameters_center_x','source_parameters_center_y',
                   'source_parameters_R_sersic']

    y_test_doppel = []
    y_pred_doppel = []
    std_pred_doppel = []
    prec_pred_doppel = []

    for i,d in enumerate(doppel_name_list):

        predictor = NetworkPredictions(weights_path_list[i],
            path_to_model_norms=norms_path,
            learning_params=learning_params,loss_type=loss_type,
            model_type='xresnet34',norm_type='lognorm')
        y_test_d, y_pred_d, std_pred_d, prec_pred_d = (
            predictor.gen_doppel_predictions(doppel_image_folder,[d],
                samples=False))
        y_test_doppel.append(y_test_d)
        y_pred_doppel.append(y_pred_d)
        std_pred_doppel.append(std_pred_d)
        prec_pred_doppel.append(prec_pred_d)
        
    # assumes y_test_d was returned with an extra dimension I think?
    y_test_doppel = np.concatenate(y_test_doppel)
    y_pred_doppel = np.concatenate(y_pred_doppel)
    std_pred_doppel = np.concatenate(std_pred_doppel)
    prec_pred_doppel = np.concatenate(prec_pred_doppel)

    return y_test_doppel, y_pred_doppel, std_pred_doppel, prec_pred_doppel


def generate_loss_curves(weights_folder,n_epochs,norm_path,test_set_folder,
    n_images):
    """
    Args:
        weights_folder (string): folder containing .h5 weights files 
            from training run
        n_epochs (int): how many epochs used for training
        norm_path (string): where norms.csv lives
        test_set_folder ([string]): folder containing paltas output for
            y_test/image pairs 
        n_images (int): # images to use for test set
    """

    learning_params = ['main_deflector_parameters_theta_E','main_deflector_parameters_gamma1',
                   'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
                   'main_deflector_parameters_e1','main_deflector_parameters_e2',
                   'main_deflector_parameters_center_x','main_deflector_parameters_center_y',
                   'source_parameters_center_x','source_parameters_center_y',
                   'source_parameters_R_sersic']
    
    # retrieve test set image/y_test pairs
    tf_path = test_set_folder+'data.tfrecord'
    tf_dataset_test = dataset_generation.generate_tf_dataset(tf_path,
        learning_params,n_images,1,norm_images=False,
        log_norm_images=True,kwargs_detector=None,
        input_norm_path=norm_path,log_learning_params=[])

    # only go through 1st iteration and grab first n_image pairs
    for t in tf_dataset_test:
        batch = t
        break

    images_list = batch[0].numpy()
    y_test_list = batch[1].numpy()

    # for each weight file, compute loss for each individual image



def generate_sequential_neg_log_pdf_curves(broad_weights_folder,broad_n_epochs,
    seq_weights_path_list,seq_n_epochs,norm_path,test_set_tfrecord=None,
    test_set_tfrecord_list=None,im_indices=None,loss_type='diag'):
    """
    Args:
        broad_weights_folder (string): folder containing .h5 weights files 
            from broad training run
        broad_n_epochs (int): how many epochs used for broad training
        seq_weights_path_list ([string]): list of folders, one for each test set 
            image, where sequential .h5 weights files live. 

            NOTE: length of seq_weights_path_list determines how many images
            the loss curve is computed for!!!

        seq_n_epochs (int)
        norm_path (string): where norms.csv lives
        test_set_tfrecord (string): If not None, assumes shifted set format 
            (all test set ims in one .tfrecrod file)
        test_set_tfrecord_list ([string]): If not None, assumes doppel set format
            (each test set im in a separate .tfrecord file), list of tfrecord 
            paths containing y_test/image pairs 
        im_indices ([int]): if specified, only certain images are evaluated. 
            put in the image indices that match the sequential weights!
        
    """

    learning_params = ['main_deflector_parameters_theta_E','main_deflector_parameters_gamma1',
                   'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
                   'main_deflector_parameters_e1','main_deflector_parameters_e2',
                   'main_deflector_parameters_center_x','main_deflector_parameters_center_y',
                   'source_parameters_center_x','source_parameters_center_y',
                   'source_parameters_R_sersic']

    # create image, y_test for each test set image
    y_test_list = []
    image_list = []
    if test_set_tfrecord is not None:
        # HARDCODED FOR A NARROW SET OF SIZE 100
        tf_dataset_test = dataset_generation.generate_tf_dataset(test_set_tfrecord,
                    learning_params,100,1,norm_images=False,
                    log_norm_images=True,kwargs_detector=None,
                    input_norm_path=norm_path,log_learning_params=[],shuffle=False)
        
        # with batch size of 100, should be exactly one batch in tf_dataset_test
        for t in tf_dataset_test:
            batch = t
        images = batch[0].numpy()
        y_test_d = batch[1].numpy()

        image_list = images
        y_test_list = y_test_d
        

    elif test_set_tfrecord_list is not None:
        for tf_path in test_set_tfrecord_list:

                tf_dataset_test = dataset_generation.generate_tf_dataset(tf_path,
                    learning_params,3,1,norm_images=False,
                    log_norm_images=True,kwargs_detector=None,
                    input_norm_path=norm_path,log_learning_params=[])

                # only one thing in tf_dataset_test since individual doppelgangers
                for t in tf_dataset_test:
                    batch = t
                images = batch[0].numpy()
                y_test_d = batch[1].numpy()

                image_list.append(images)
                y_test_list.append(y_test_d)

    # initialize array to store loss values
    # shape: (n_test_images,broad_n_epochs+seq_n_epochs)
    loss_array = np.empty((len(seq_weights_path_list),broad_n_epochs+seq_n_epochs))

    # loop through broad weights, load them in, then compute loss for single images
    weights_files = broad_weights_folder+'xresnet34_*.h5'
    broad_weights_files = np.asarray(os.popen('ls -d '+weights_files).read().split())
    for i in range(0,broad_n_epochs):
        path_to_model_weights = broad_weights_files[i]
        network_pred = NetworkPredictions(path_to_model_weights,
            path_to_model_norms=norm_path,learning_params=learning_params,
            loss_type=loss_type,model_type='xresnet34',norm_type='lognorm')
        
        # loop through each test set image & compute loss
        for t in range(0,len(seq_weights_path_list)):
            if im_indices is not None:
                idx = im_indices[t]
            else:
                idx = t
            if test_set_tfrecord is not None:
                test_image = np.asarray([image_list[idx]])
            else:
                test_image = image_list[idx]
            if i == 0:
                print('checking y_test: ', y_test_list[idx])
            l = network_pred.loss_single_image(y_test_list[idx],test_image)
            loss_array[t,i] = l

    # now loop through test set images to compute w/ sequential weights
    for t in range(0,len(seq_weights_path_list)):

        # loop through sequential weights
        weights_files = seq_weights_path_list[t]+'xresnet34_*.h5'
        seq_weights_files = np.asarray(os.popen('ls -d '+weights_files).read().split())
        for s in range(0,seq_n_epochs):
            path_to_model_weights = seq_weights_files[s]

            network_pred = NetworkPredictions(path_to_model_weights,
                path_to_model_norms=norm_path,learning_params=learning_params,
                loss_type=loss_type,model_type='xresnet34',norm_type='lognorm')
            if im_indices is not None:
                idx = im_indices[t]
            else:
                idx = t
            if test_set_tfrecord is not None:
                test_image = np.asarray([image_list[idx]])
            else:
                test_image = image_list[idx]
            l = network_pred.loss_single_image(y_test_list[idx],test_image)
            loss_array[t,s+broad_n_epochs] = l

    return loss_array

