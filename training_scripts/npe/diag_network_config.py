import os
import random

####################
# Things that change
####################

training_folder = '/scratch/users/sydney3/paper_results/broad_training/'
write_folder = '/scratch/users/sydney3/paper_results/broad_training/diag_no_R_src/'
num_training_folders = 10

input_norm_path = (write_folder + 'norms.csv')

# loops thru training set
n_epochs = 100
# Steps Per Decay
steps_per_decay = 1e3

# A string with which loss function to use.
loss_function = 'diag'

# Whether or not to normalize the images by the standard deviation
norm_images = False
log_norm_images = True

# Where to save the model weights
model_weights = (write_folder +
    'xresnet34_{epoch:03d}-{val_loss:.2f}.h5')

model_weights_init = None
#model_weights_init = (write_folder + 'xresnet34_165--19.18_last.h5')

# The learning rate for the model
learning_rate = 5e-4
#learning_rate = 5e-3*(0.98**(165*5e5/(512*1e3)))
# Whether or not to use random rotation of the input images

random_rotation = True

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
        'source_parameters_center_y']
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
