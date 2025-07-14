#!/bin/bash
#SBATCH --job-name=thesis               # Job name
#SBATCH --time=16:00:00                 # Time limit (hh:mm:ss)
#SBATCH --partition=general             # Partition/queue name
#SBATCH --cpus-per-task=2               # CPUs per task 
#SBATCH --mem-per-cpu=2G


source /cluster/home/prasadm/miniconda3/etc/profile.d/conda.sh
conda activate lisa_env
echo "cpus-per-task=2, mem-per-cpu=2G, 'workers': 2"
echo "Tobs = YRSID_SI/3, dist = 20 * PC_SI * 1e9, nperseg = 5000"
python /cluster/home/prasadm/LISA_Master_Thesis/differential_evolution/diff_evolution.py