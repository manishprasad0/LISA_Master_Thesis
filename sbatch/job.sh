#!/bin/bash
#SBATCH --job-name=thesis               # Job name
#SBATCH --account=root
#SBATCH --time=1-00:00:00            
#SBATCH --partition=normal           
#SBATCH --cpus-per-task=32          
#SBATCH --mem-per-cpu=3G       # The fastest. see the output slurm-51901. It went to 1500 iters

source /users/prasadm/miniconda3/etc/profile.d/conda.sh
conda activate lisa_env

echo "cpus-per-task=32, mem-per-cpu=3G, workers=3"
echo "used old diff_evol kwargs settings from github"
echo "'maxiter': 100 for testing to see how long it takes"
echo "nperseg = 5000, reduced observation time to 1 month"
echo "reduced maxiter to 100 to see speed"

python -u /users/prasadm/LISA_Master_Thesis/differential_evolution/diff_evolution.py