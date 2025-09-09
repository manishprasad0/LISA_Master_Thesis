#!/bin/bash
#SBATCH --job-name=thesis               # Job name
#SBATCH --account=root
#SBATCH --time=1-00:00:00            
#SBATCH --partition=normal           
#SBATCH --cpus-per-task=32          
#SBATCH --mem-per-cpu=3G       # The fastest. see the output slurm-51901. It went to 1500 iters

source /users/prasadm/miniconda3/etc/profile.d/conda.sh
conda activate lisa_env

echo "RUN 3 : FIXED T_REF"
echo "maxiter = 1500"
echo "DE_find_tref.py = 1 month, 60 deg, 20 deg"
echo "actually stefan's kwargs"
echo "cpus-per-task=32, mem-per-cpu=3G, workers=3"
echo "nperseg = 1414, reduced observation time to 1 month" 

python -u /users/prasadm/LISA_Master_Thesis/differential_evolution/DE_full.py