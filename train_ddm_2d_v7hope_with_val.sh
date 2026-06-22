#!/bin/bash
#SBATCH --job-name=train_ddm_2d_v7hope_val
#SBATCH --output=logs/train_ddm_2d_v7hope_val_%j.out
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.ddm.train_ddm
