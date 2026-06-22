#!/bin/bash
#SBATCH --job-name=active_units
#SBATCH --output=logs/active_units_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u compute_active_units.py
