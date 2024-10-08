#!/bin/bash

#SBATCH -J train_sl
#SBATCH -p gpu
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 01:00:00
#SBATCH --mem 8000
#SBATCH -o /home/users/sydney3/slurm_out/seq_train_%j.out
#SBATCH -e /home/users/sydney3/slurm_out/seq_train_%j.err
#SBATCH -C GPU_BRD:GEFORCE

module load py-tensorflow/2.9.1_py39
cd /home/users/sydney3/paltas/paltas/Analysis
python3 train_model.py /home/users/sydney3/deep-lens-modeling/paper/sequential_training/network_seq_config.py --h5