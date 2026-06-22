#!/bin/bash
#SBATCH --job-name=ablation_winit
#SBATCH --output=logs/ablation_winit_%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u ablation_winit.py \
    --checkpoint      results/ddm_w_3d/ddm_w_3d_v29_best.pth \
    --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
    --n_passes        10 \
    --seed            0
