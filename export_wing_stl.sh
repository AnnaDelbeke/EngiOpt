#!/bin/bash
#SBATCH --job-name=export_wing_stl
#SBATCH --output=logs/export_wing_stl_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u export_wing_stl.py \
    --case_idx 0 \
    --chord_m  1.0 \
    --out      wing_generated.stl
