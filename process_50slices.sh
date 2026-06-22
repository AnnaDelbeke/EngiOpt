#!/bin/bash
#SBATCH --job-name=process_50slices
#SBATCH --output=logs/process_50slices_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=16

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt/Wing_TL

python -u -m wing_tl.data_processing.run.process_new_dataset \
    --dataset_root data/raw/new_dataset \
    --save_dir     data/processed \
    --n_span_slices 50 \
    --n_interp 192 \
    --workers 16 \
    --save_prefix new_dataset_50slices
