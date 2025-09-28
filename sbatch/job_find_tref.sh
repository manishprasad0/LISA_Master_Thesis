#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --account=root
#SBATCH --time=1-00:00:00            
#SBATCH --partition=normal           
#SBATCH --cpus-per-task=32          
#SBATCH --mem-per-cpu=3G       # The fastest. see the output slurm-51901. It went to 1500 iters

source /users/prasadm/miniconda3/etc/profile.d/conda.sh
conda activate lisa_env

echo "RUN 1 : FINDING T_REF LOWER SNR = 932"
echo "Time before merger = 10 hours, Tobs = 1.5 months"
echo "maxiter = 100, nperseg = 1414"
echo "diff_evolution.py = 1 month, 60 deg, 20 deg"
echo "recombination: 0.6"
echo "mutation: (0.5, 0.8)"
echo "cpus-per-task=32, mem-per-cpu=3G, workers=3"
echo "dist = 20 GPc, phi_ref = 0.577, psi = 0.217"

python -u /users/prasadm/LISA_Master_Thesis/differential_evolution/DE_find_tref.py 
#> /users/prasadm/LISA_Master_Thesis/slurm_differential_evolution/slurm-$(date +'%d-%m-%Y_%H-%M-%S').out