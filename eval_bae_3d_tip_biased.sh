#!/bin/bash
#SBATCH --job-name=eval_bae_tip_biased
#SBATCH --output=logs/eval_bae_tip_biased_%j.out
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u -m engiopt.bezier_ae.evaluate_bezier_ae_3d \
    --checkpoint results/bezier_ae_3d/run_047/models/bezier_ae_3d_best.pt \
    --run_dir    results/bezier_ae_3d/run_047 \
    --slices_pkl Wing_TL/data/processed/new_dataset_tip_biased_slices.pkl \
    --n_examples 5
