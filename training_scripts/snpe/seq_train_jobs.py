import os


indices = range(0,14)


base_config_file = '/home/users/sydney3/deep-lens-modeling/paper/sequential_training/base_network_seq_config.py'
base_bash_file = '/home/users/sydney3/deep-lens-modeling/paper/sequential_training/base_Sherlock_train_network.sh'

for i in indices:
        # get sequential proposal from sequential config file!
        seq_proposal_filepath = ('/scratch/users/sydney3/paper_results/sequential_training/data_test_set/'+
                'data_configs/config_data%03d.py'%(i))
        with open(seq_proposal_filepath) as file:
                lines = [line.rstrip() for line in file]
                proposal_means = lines[11][10:]+lines[12][5:]
                proposal_scatters = lines[13][13:]+lines[14][5:]

        write_path = '/scratch/users/sydney3/paper_results/sequential_training/data_test_set/data%03d/'%(i)
        # copy base_network_seq_config.py to scratch & make the right changes
        with open(base_config_file) as file:
                lines = [line.rstrip() for line in file]
                lines[8] = 'training_folder = \'/scratch/users/sydney3/paper_results/sequential_training/data_test_set/data%03d/\''%(i)
                lines[9] = 'write_folder = \'/scratch/users/sydney3/paper_results/sequential_training/data_test_set/data%03d/\''%(i)

                # TODO: write in APT proposal here
                lines[13] = 'proposal_means = np.asarray('+proposal_means+')'
                lines[14] = 'proposal_prec = np.linalg.inv(np.diag(np.asarray('+proposal_scatters+')**2))'

        with open(write_path+'network_config.py', 'w') as f:
                for line in lines:
                        f.write(f"{line}\n")

        # copy base bash script
        with open(base_bash_file) as file:
                lines = [line.rstrip() for line in file]
                lines[15] = 'python3 train_model.py '+write_path+'network_config.py --h5'

        with open(write_path+'train_net.sh', 'w') as f:
                for line in lines:
                        f.write(f"{line}\n")

        # submit the job!
        os.system('sbatch '+write_path+'train_net.sh')