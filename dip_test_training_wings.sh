#!/bin/bash
#SBATCH --job-name=dip_test_train
#SBATCH --output=logs/dip_test_training_wings_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u run_dip_test_training_wings.py
