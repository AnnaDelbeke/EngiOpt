#!/bin/bash
#SBATCH --job-name=process_custom_etas
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=normal.4h
#SBATCH --output=logs/process_custom_etas_%j.out

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
source .venv/bin/activate

DATASET_ROOT=/cluster/home/adelbeke/EngiOpt/Wing_TL/data/raw/new_dataset

python -u -m wing_tl.data_processing.run.process_new_dataset \
    --dataset_root "$DATASET_ROOT" \
    --save_dir     Wing_TL/data/processed \
    --n_span_slices 15 \
    --n_interp     192 \
    --workers      32 \
    --save_prefix  new_dataset_custom_etas \
    --custom_etas  0.01 0.36 0.62 0.94 1.25 1.45 1.65 1.85 2.10 2.20 2.30 2.35 2.40 2.45 2.50
