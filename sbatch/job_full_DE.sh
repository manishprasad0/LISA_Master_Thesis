#!/bin/bash
#SBATCH --job-name=thesis               # Job name
#SBATCH --account=root
#SBATCH --time=1-00:00:00            
#SBATCH --partition=normal           
#SBATCH --cpus-per-task=32          
#SBATCH --mem-per-cpu=3G       # The fastest. see the output slurm-51901. It went to 1500 iters

source /users/prasadm/miniconda3/etc/profile.d/conda.sh
conda activate lisa_env

echo "maxiter = 1500"
echo "DE_full.py"

python -u /users/prasadm/LISA_Master_Thesis/differential_evolution/DE_full.py