#!/bin/bash
#SBATCH --job-name=plot_abl_midspan
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/plot_ablation_midspan_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1

python -u plot_ablation_midspan.py \
    --ablation_dir    results/ddm_w_3d_ablation_v29 \
    --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
    --span_idx        7 \
    --wing_idx        0 \
    --out             results/ablation_v29_midspan.png
