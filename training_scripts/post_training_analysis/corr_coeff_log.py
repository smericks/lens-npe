# re-calculate network training log files from a training run by loading in network weights
# and computing metrics on validation set
import numpy as np
import os
import tensorflow as tf
from paltas.Analysis import loss_functions, conv_models, dataset_generation
import pandas as pd
import time

debug = True
weights_files = '/scratch/users/sydney3/july31_2023_lognorm/xresnet34_*last.h5'
tfr_val_path = '/scratch/users/sydney3/july31_2023/validate/data.tfrecord'
folder_indices = range(0,500)
input_norm_path = '/scratch/users/sydney3/july31_2023_lognorm/norms.csv'
write_path = '/scratch/users/sydney3/july31_2023_lognorm/corr_coeff_log.csv'
N_val = 5000

model_weights_list = np.asarray(os.popen('ls -d '+weights_files).read().split())

img_size = (80,80,1)
random_seed = 2
batch_size = 512
flip_pairs = None
norm_images = False
log_norm_images = True
learning_params = ['main_deflector_parameters_theta_E','main_deflector_parameters_gamma1',
                   'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
                   'main_deflector_parameters_e1','main_deflector_parameters_e2',
                   'main_deflector_parameters_center_x','main_deflector_parameters_center_y']
learning_params_print = [r'$\theta_\mathrm{E}$',r'$\gamma_1$',r'$\gamma_2$',r'$\gamma_\mathrm{lens}$',r'$e_1$',
                         r'$e_2$',r'$x_\mathrm{lens}$',r'$y_\mathrm{lens}$']
log_learning_params = []
num_params = len(learning_params+log_learning_params)
loss_type = 'full'
model_type = 'xresnet34'
tf.random.set_seed(random_seed)

corr_coeffs = np.empty((len(model_weights_list),num_params))

if loss_type == 'full':
    num_outputs = num_params + int(num_params*(num_params+1)/2)
    loss_func = loss_functions.FullCovarianceLoss(num_params)

def compute_cc(y_test,output):
    y_pred, _, _ = loss_func.convert_output(output)
    cc_row = np.empty(num_params)
    for i in range(num_params):
        cc_row[i] = np.corrcoef(y_pred[:,i],y_test[:,i])[0,1]

    return cc_row

tf_dataset_val = dataset_generation.generate_tf_dataset(tfr_val_path,learning_params,
            N_val,1,norm_images=norm_images,log_norm_images=log_norm_images,
            kwargs_detector=None,input_norm_path=input_norm_path,
            log_learning_params=log_learning_params,shuffle=False)

# should only iterate through once
for batch in tf_dataset_val:
    images_val = batch[0].numpy()
    y_test_val = batch[1].numpy()

count = 0

for i,model_weights in enumerate(model_weights_list):

    model = conv_models.build_xresnet34(img_size,num_outputs)
    model.load_weights(model_weights,by_name=True,skip_mismatch=True)
    output_val = model.predict(images_val)

    corr_coeffs[i] = compute_cc(y_test_val,output_val)

    if debug and count > 2:
        break
    count += 1

df = pd.DataFrame(corr_coeffs, columns = learning_params)
df.to_csv(write_path)