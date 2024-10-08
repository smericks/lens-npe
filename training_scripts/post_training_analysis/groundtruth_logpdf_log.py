from network_predictions import generate_sequential_neg_log_pdf_curves
import numpy as np

debug = False
#narrow_idx = 0
doppel_idx = 11
broad_weights_folder = '/scratch/users/sydney3/paper_results/broad_training/full_no_R_src/'
broad_n_epochs = 46
norm_path = '/scratch/users/sydney3/paper_results/broad_training/diag_no_R_src/norms.csv'
seq_weights_path_list = [
        '/scratch/users/sydney3/full_cov_snpe/doppelganger_test_set/doppel011/'
]
#im_indices = [narrow_idx]
seq_n_epochs = 41

# for narrow, single file. for doppels, separate files
#test_set_tfrecord = '/scratch/users/sydney3/paper_results/sequential_training/shifted_test_set/data.tfrecord' 
test_set_tfrecord_list = ['/scratch/users/sydney3/full_cov_snpe/doppelganger_test_set/doppel011/doppel_data.tfrecord']

log_array = generate_sequential_neg_log_pdf_curves(broad_weights_folder,broad_n_epochs,
    seq_weights_path_list,seq_n_epochs,norm_path,test_set_tfrecord_list=test_set_tfrecord_list,im_indices=None,
    loss_type='full')

np.save(seq_weights_path_list[0]+'seq_loss_curve.npy',log_array)