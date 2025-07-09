#!/bin/bash
#SBATCH --job-name=lisa_master_thesis   # Job name
#SBATCH --time=10:00:00                 # Time limit (hh:mm:ss)
#SBATCH --partition=general             # Partition/queue name
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10G


source /cluster/home/prasadm/miniconda3/etc/profile.d/conda.sh
conda activate lisa_env
echo "cpus-per-task=16 and mem-per-cpu=10G"
python /cluster/home/prasadm/LISA_Master_Thesis/differential_evolution/diff_evolution.py