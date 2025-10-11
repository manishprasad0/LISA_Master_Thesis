#!/bin/bash
#SBATCH --job-name=thesis               # Job name
#SBATCH --account=root
#SBATCH --time=1-00:00:00            
#SBATCH --partition=normal           
#SBATCH --cpus-per-task=1          
#SBATCH --mem-per-cpu=3G

source /users/prasadm/miniconda3/etc/profile.d/conda.sh
conda activate lisa_gpu

echo "maxiter = 1500"
echo "DE_full_GPU.py"

python -u /users/prasadm/LISA_Master_Thesis/differential_evolution/DE_full_GPU.py