import os

os.system('sbatch Sherlock_generate_validation.sh')
for i in range(0,100):
    os.system('sbatch Sherlock_generate_training.sh %d'%(i))