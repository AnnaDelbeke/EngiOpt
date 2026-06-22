#!/bin/bash
#SBATCH --job-name=find_worst_tip
#SBATCH --output=logs/find_worst_tip_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=8G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u find_worst_tip_wing.py
