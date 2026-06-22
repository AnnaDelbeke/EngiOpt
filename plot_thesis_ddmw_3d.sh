#!/bin/bash
#SBATCH --job-name=plot_thesis_ddmw_3d
#SBATCH --output=logs/plot_thesis_ddmw_3d_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G

set -e
cd /cluster/home/adelbeke/EngiOpt
source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate

mkdir -p logs

python -m engiopt.ddm.ddm_w.plot_thesis_ddmw_3d
