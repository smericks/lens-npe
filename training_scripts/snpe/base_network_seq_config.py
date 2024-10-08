import os
import random
import numpy as np

####################
# Things that change
####################

training_folder = '/scratch/users/sydney3/paper_results/sequential_training/shifted_test_set/narrow000/'
write_folder = '/scratch/users/sydney3/paper_results/sequential_training/shifted_test_set/narrow000/'
num_training_folders = 1

# APT proposal
proposal_means = np.asarray([ 0.567,  0.027,  0.167,  1.911,  0.173,  0.037, -0.018,  0.034, -0.035,  0.011,  0.728])
proposal_prec = np.linalg.inv(np.diag(np.asarray([0.023, 0.039, 0.054, 0.11 , 0.093, 0.121, 0.009, 0.008, 0.007, 0.007, 0.284])**2))
# APT prior
prior_means = np.asarray([0.8,0.,0.,2.0,0.,0.,0.,0.,0.,0.,0.5])
prior_prec = np.linalg.inv(np.diag(np.asarray([0.15,0.12,0.12,0.2,0.2,0.2,0.07,0.07,0.1,0.1,0.5])**2))

input_norm_path = ('/scratch/users/sydney3/paper_results/broad_training/' + 'norms.csv')

# loops thru training set
n_epochs = 50
# Steps Per Decay
steps_per_decay = 100

# A string with which loss function to use.
loss_function = 'diagapt'

# Whether or not to normalize the images by the standard deviation
norm_images = False
log_norm_images = True

# Where to save the model weights
model_weights = (write_folder +
    'xresnet34_{epoch:03d}-{val_loss:.2f}.h5')

#model_weights_init = None
model_weights_init = ('/scratch/users/sydney3/paper_results/broad_training/'+
                      'xresnet34_053--14.10_best.h5')

# The learning rate for the model
#learning_rate = 5e-4
learning_rate = 5e-4*(0.98**(53*5e5/(512*1e3)))

# Whether or not to use random rotation of the input images
#  NEEDS TO BE TURNED OFF FOR SEQUENTIAL
random_rotation = False

# Save training results to .csv file
csv_path = (write_folder + 'log.csv')

##########################
# Things that don't change
##########################
batch_size = 512
img_size = (80,80,1)
learning_params = ['main_deflector_parameters_theta_E',
        'main_deflector_parameters_gamma1','main_deflector_parameters_gamma2',
        'main_deflector_parameters_gamma','main_deflector_parameters_e1',
        'main_deflector_parameters_e2','main_deflector_parameters_center_x',
        'main_deflector_parameters_center_y','source_parameters_center_x',
        'source_parameters_center_y','source_parameters_R_sersic']
flip_pairs = None
weight_terms = None

# prep training / validation paths
folder_indices = range(0,num_training_folders)
npy_folders_train = [
        (training_folder+'train_%d/'%(i)) for i in folder_indices]
tfr_train_paths = [
        os.path.join(path,'data.tfrecord') for path in npy_folders_train]
npy_folder_val = (training_folder+'validate/')
tfr_val_path = os.path.join(npy_folder_val,'data.tfrecord')
metadata_paths_train = [
        os.path.join(path,'metadata.csv') for path in npy_folders_train]
metadata_path_val = os.path.join(npy_folder_val,'metadata.csv')
# The detector kwargs to use for on-the-fly noise generation
kwargs_detector = None
# A string specifying which model to use
model_type = 'xresnet34'
# A string specifying which optimizer to use
optimizer = 'Adam'
