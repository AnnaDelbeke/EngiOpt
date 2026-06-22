#!/bin/bash
#SBATCH --job-name=plot_ablation_geom_only
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/plot_ablation_geom_only_%j.out

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

python plot_ablation_midspan.py \
    --ablation_dir    results/ddm_w_3d_ablation_geom_only \
    --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d_geom_only/lvae_3d_v29_geom_only_best.pth \
    --span_idx        7 \
    --wing_idx        0 \
    --out             results/ablation_geom_only_midspan.png
