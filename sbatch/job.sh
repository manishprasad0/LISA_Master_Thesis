#!/bin/bash
#SBATCH --job-name=thesis               # Job name
#SBATCH --time=40:00:00                 # Time limit (hh:mm:ss)
#SBATCH --partition=general             # Partition/queue name
#SBATCH --cpus-per-task=3               # CPUs per task 
#SBATCH --mem-per-cpu=3G

source /cluster/home/prasadm/miniconda3/etc/profile.d/conda.sh
conda activate lisa_env
echo "hours before merger = 10, width = 20, initial run"
python /cluster/home/prasadm/LISA_Master_Thesis/differential_evolution/diff_evolution.py