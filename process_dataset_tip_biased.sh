#!/bin/bash
#SBATCH --job-name=process_tip_biased
#SBATCH --output=logs/process_tip_biased_%j.out
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=16

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u -m wing_tl.data_processing.run.process_new_dataset \
    --dataset_root Wing_TL/data/raw/new_dataset \
    --save_dir     Wing_TL/data/processed \
    --n_span_slices 15 \
    --n_interp 192 \
    --workers 16 \
    --save_prefix new_dataset_tip_biased \
    --tip_biased
