#!/bin/bash
#SBATCH --job-name=regen_wing0_gt
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/regen_wing0_gt_%j.out

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

python3 -m engiopt.ddm.regenerate_wing0_with_gt
