#!/bin/bash
#SBATCH -A <your_slurm_account>
#SBATCH -p u22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=$HOME_ROOT/IRASTE/DAY_2/slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

echo "========================================"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "HOSTNAME: $(hostname)"
echo "========================================"

nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

python $HOME_ROOT/IRASTE/DAY_2/run_florence2.py
