#!/bin/bash
#SBATCH --job-name=eval_ddm_v7
#SBATCH --output=logs/eval_ddm_v7_%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.ddm.evaluate_ddm
