#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --time=40:00:00      
#SBATCH --partition=general           
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=3G

source /cluster/home/prasadm/miniconda3/etc/profile.d/conda.sh
conda activate lisa_env

echo "RUN 1 : FINDING T_REF"
echo "Time before merger = 14 hours"
echo "maxiter = 100"
echo "DE_find_tref.py = 1 month, 60 deg, 20 deg"
echo "recombination: 0.6"
echo "mutation: (0.5, 0.8)"
echo "cpus-per-task=3, mem-per-cpu=3G, workers=3"
echo "nperseg = 1414, increased observation time to 1.5 month"

python -u /cluster/home/prasadm/LISA_Master_Thesis/differential_evolution/DE_find_tref.py