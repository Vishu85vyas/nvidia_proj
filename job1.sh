#!/bin/bash
#SBATCH --job-name=VQE-CudaQ
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --partition=debug

echo "=== SLURM Job Info ==="
echo "Node       : $(hostname)"
echo "Job ID     : ${SLURM_JOB_ID:-manual}"
echo "User       : $USER"
echo "Start Time : $(date)"
echo "======================"

# >>> Properly initialize conda <<<
source ~/miniconda3/etc/profile.d/conda.sh
conda activate moldysim || { echo "Failed to activate conda env"; exit 1; }

# Go to working directory (optional)
cd /home/$USER || exit

# Run the VQE script
echo "Running vqe.py..."
#python vqe4.py
python vqe4.py

echo "=== Job Completed at $(date) ==="

