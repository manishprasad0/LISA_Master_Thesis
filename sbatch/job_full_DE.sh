#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --time=40:00:00      
#SBATCH --partition=general           
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=3G

source /cluster/home/prasadm/miniconda3/etc/profile.d/conda.sh
conda activate lisa_env

echo "RUN 1 : FULL DE"
echo "maxiter = 1500"
echo "DE_full.py = 1 month, 60 deg, 20 deg"
echo "actually stefan's kwargs"
echo "cpus-per-task=32, mem-per-cpu=3G, workers=3"
echo "nperseg = 1414, increased observation time to 1.5 months"

python -u /cluster/home/prasadm/LISA_Master_Thesis/differential_evolution/DE_full.py