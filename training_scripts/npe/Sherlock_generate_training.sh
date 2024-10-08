#!/bin/bash

#SBATCH -J gen_ts
#SBATCH -p kipac,normal,hns
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -t 07:00:00
#SBATCH --mem 8000
#SBATCH --error /home/users/sydney3/slurm_out/gen_ts_%j.err
#SBATCH --output /home/users/sydney3/slurm_out/gen_ts_%j.out

module load python/3.9.0
module load system ruse
cd /home/users/sydney3/paltas/paltas
# argument 1 is the training folder index
python3 generate.py /home/users/sydney3/deep-lens-modeling/paper/broad_training/training_config_broad.py /scratch/users/sydney3/paper_results/broad_training/train_$1 --n 50000 --tf_record --h5